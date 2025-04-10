using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace register_packager;

unsafe class Memory
{
    [DllImport("ucrtbase", CallingConvention = CallingConvention.Cdecl, EntryPoint = "malloc")]
    public static extern void* Alloc(nint size);

    [DllImport("ucrtbase", CallingConvention = CallingConvention.Cdecl, EntryPoint = "free")]
    public static extern void Free(void* pointer);

    static MemoryProvider Provider = Avx2.IsSupported ? new AVX2MemoryProvied() : new DefaultMemoryProvider();

    public static void Copy(void* source, void* destination, int length) => Provider.Copy(source, destination, length);

    abstract class MemoryProvider
    {
        public void Copy(void* source, void* destination, int length) => Copy((byte*)source, (byte*)destination, length);
        public abstract void Copy(byte* source, byte* destination, int length);
    }

    class DefaultMemoryProvider : MemoryProvider
    {
        public override unsafe void Copy(byte* source, byte* destination, int length) => Buffer.MemoryCopy(source, destination, length, length);
    }

    class AVX2MemoryProvied : MemoryProvider
    {
        public override unsafe void Copy(byte* source, byte* destination, int length)
        {
            const int BlockSize = 32;

            int i = 0;
            int lastBlockIndex = length - BlockSize;
            for (; i <= lastBlockIndex; i += BlockSize)
            {
                var vector = Avx.LoadVector256(source + i);
                Avx.Store(destination + i, vector);
            }

            for (; i < length; i++)
                source[i] = destination[i];
        }
    }
}

public unsafe class Algorithm : IDisposable
{
    Algorithm(int maxLimit, int[] registersArray)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLimit);
        ArgumentOutOfRangeException.ThrowIfZero(registersArray.Length);

        this.maxLimit = maxLimit;
        this.registersArray = registersArray;

        registersHandle = GCHandle.Alloc(registersArray, GCHandleType.Pinned);
        registers = (int*)registersHandle.AddrOfPinnedObject();
    }

    int maxLimit;
    int[] registersArray;
    GCHandle registersHandle;
    int* registers;

    int[][] InstanceSolve()
    {
        var root = Chunk(maxLimit, registersArray).Next;
        ArgumentNullException.ThrowIfNull(root);
        var node = JoinRecursive(maxLimit, GetNumberWithZeros(maxLimit), root, false);
        return GetChunks(node).ToArray();
    }

    int GetNumberWithZeros(int x) => (int)Math.Pow(10, (int)Math.Floor(Math.Log10(x)) + 1);

    IEnumerable<int[]> GetChunks(Node node)
    {
        var current = node;
        while (current is not null)
        {
            if (current.Registers.Length != 0)
            {
                yield return current.Registers.ToArray(this);
            }
            current = current.Next;
        }
    }

    class Node
    {
        [AllowNull] public Array Registers;
        public Node? Next;
    }

    class Array
    {
        Array(int start, int end) => (Start, End) = (start, end);

        public int Start;
        public int End;

        public int Length => End - Start;

        public static Array Empty = FromRange(0, 0);

        public int[] ToArray(Algorithm algorithm)
        {
            var length = Length;
            var offset = Start;

            var array = new int[length];
            fixed (int* arrayPointer = array)
                Memory.Copy(algorithm.registers + offset, arrayPointer, length * sizeof(int));

            return array;
        }

        public static Array FromRange(int start, int end) => new(start, end);
    }

    Node JoinRecursive(int maxLimit, int decimalOrderMaxLimit, Node root, bool rearrange)
    {
        var node = root;
        while (node is not null)
        {
            var current = node.Registers;
            if (node.Next is not null)
            {
                var follow = node.Next.Registers;
                if (follow.Length == 0)
                {
                    node = node.Next;
                    continue;
                }
                var heightRest = CalculateHeight(node.Next.Next);
                var garbageRest = CalculateGarbage(node.Next.Next);
                var min = CalculateGarbage(current, follow) + garbageRest + decimalOrderMaxLimit * heightRest;
                var prefer = node;
                foreach (var (trimLeft, joinRight) in CombineWithLowerGarbageThanSource(current, follow))
                {
                    if (trimLeft.Length != 0 && ExcessLimit(maxLimit, joinRight, out var taken, out var rest))
                    {
                        if (rearrange)
                        {
                            continue;
                        }
                        var next = JoinRecursive(maxLimit, decimalOrderMaxLimit, CreateNodeWithoutEmptyRegisters(Array.Empty, rest, node.Next.Next), true);
                        if (CalculateHeight(next) <= CalculateHeight(node.Next.Next))
                        {
                            var garbage = CalculateGarbage(trimLeft, taken) + CalculateGarbage(next) + decimalOrderMaxLimit * CalculateHeight(next);
                            if (garbage < min)
                            {
                                min = garbage;
                                prefer = CreateNodeWithoutEmptyRegisters(trimLeft, taken, next);
                            }
                        }
                    }
                    if (!ExcessLimit(maxLimit, joinRight))
                    {
                        var garbage = CalculateGarbage(trimLeft, joinRight) + garbageRest + decimalOrderMaxLimit * (heightRest - (trimLeft.Length == 0 ? 1 : 0));
                        if (garbage < min)
                        {
                            min = garbage;
                            prefer = CreateNodeWithoutEmptyRegisters(trimLeft, joinRight, node.Next.Next);
                        }
                    }
                }
                node.Registers = prefer.Registers;
                node.Next = prefer.Next;
            }
            else
            {
                return root;
            }
            node = node.Next;
        }
        return root;
    }

    Node CreateNodeWithoutEmptyRegisters(Array left, Array right, Node? rest)
    {
        if (left.Length == 0)
        {
            return new Node()
            {
                Registers = right,
                Next = rest
            };
        }
        if (right.Length == 0)
        {
            return new Node()
            {
                Registers = left,
                Next = rest
            };
        }
        return new Node()
        {
            Registers = left,
            Next = new Node()
            {
                Registers = right,
                Next = rest
            }
        };
    }

    bool ExcessLimit(int maxLimit, Array chunk, out Array taken, out Array rest)
    {
        ArgumentOutOfRangeException.ThrowIfZero(chunk.Length);

        if (ExcessLimit(maxLimit, chunk))
        {
            var index = chunk.Length - 1;
            while (index >= 0)
            {
                if (!ExcessLimit(maxLimit, chunk[..index]))
                {
                    break;
                }
                index--;
            }
            rest = chunk[index..].ToArray();
            taken = chunk[..index].ToArray();
            return true;
        }
        rest = [];
        taken = chunk;
        return false;
    }

    bool ExcessLimit(int maxLimit, Array chunk) => chunk[^1] - chunk[0] + 1 > maxLimit;

    int CalculateGarbage(Array chunk1, Array chunk2) => chunk1.Length == 0 ? CalculateGarbage(chunk2) : CalculateGarbage(chunk1) + CalculateGarbage(chunk2);

    int CalculateHeight(Node? node)
    {
        var height = 0;
        var current = node;
        while (current is not null)
        {
            if (current.Registers.Length != 0)
            {
                height++;
            }
            current = current.Next;
        }
        return height;
    }

    int CalculateGarbage(Node? node)
    {
        var garbage = 0;
        var current = node;
        while (current is not null)
        {
            garbage += CalculateGarbage(current.Registers);
            current = current.Next;
        }
        return garbage;
    }

    int CalculateGarbage(Array chunk)
    {
        ArgumentOutOfRangeException.ThrowIfZero(chunk.Length);

        var garbage = 0;
        var index = 1;  
        while (index < chunk.Length)
        {
            garbage += chunk[index] - chunk[index - 1] - 1;
            index++;
        }
        return garbage;
    }

    List<(Array TrimLeft, Array JoinRight)> CombineWithLowerGarbageThanSource(Array chunk1, Array chunk2)
    {
        List<(Array TrimLeft, Array JoinRight)> res = [];
        var min = CalculateGarbage(chunk1, chunk2);
        var concat = [.. chunk1, .. chunk2];
        for (var splitPoint = chunk1.Length - 1; splitPoint >= 0; splitPoint--)
        {
            var trimLeft = concat[..splitPoint];
            var joinRight = concat[splitPoint..];
            var garbage = CalculateGarbage(trimLeft, joinRight);
            if (garbage < min || trimLeft.Length == 0)
            {
                min = garbage;
                res.Add((trimLeft.ToArray(), joinRight.ToArray()));
            }
        }
        return res;
    }

    Node Chunk(int maxLimit, int[] registers)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLimit);
        ArgumentOutOfRangeException.ThrowIfZero(registers.Length);

        var root = new Node();
        var index = 0;
        var previous = registers[0];
        var chunkStart = 0;
        var currentLimit = 1;
        var node = root;
        while (index < registers.Length)
        {
            var current = registers[index];
            var distance = current - previous;
            currentLimit += distance;
            if (currentLimit > maxLimit)
            {
                node.Next = new Node()
                {
                    Registers = registers[chunkStart..index]
                };
                node = node.Next;
                currentLimit = 1;
                chunkStart = index;
            }
            previous = current;
            index++;
        }
        if (currentLimit != 0)
        {
            node.Next = new Node()
            {
                Registers = registers[chunkStart..index]
            };
        }
        return root;
    }

    public void Dispose() => registersHandle.Free();

    public static int[][] Solve(int maxLimit, int[] registers)
    {
        using var algorithm = new Algorithm(maxLimit, registers);
        return algorithm.InstanceSolve();
    }
}


/*
public class Algorithm
{
    public static int[][] Solve(int maxLimit, int[] registers)
    {
        using var algorithmInstance = new AlgorithmInstance(maxLimit, registers);
        return algorithmInstance.Solve();
    }    
}

unsafe class AlgorithmInstance : IDisposable
{
    public AlgorithmInstance(int maxLimit, int[] registersArray)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLimit);
        ArgumentOutOfRangeException.ThrowIfZero(registersArray.Length);

        this.maxLimit = maxLimit;
        this.registersArray = registersArray;

        registersHandle = GCHandle.Alloc(registersArray, GCHandleType.Pinned);
        registers = (int*)registersHandle.AddrOfPinnedObject();
    }

    int maxLimit;
    int[] registersArray;
    GCHandle registersHandle;
    int* registers;

    public int[][] Solve()
    {
        var root = Chunk().Next;
        ArgumentNullException.ThrowIfNull(root);
        var node = JoinRecursive(maxLimit, GetNumberWithZeros(maxLimit), root, false);
        return GetChunks(node).ToArray();
    }

    int GetNumberWithZeros(int x) => (int)Math.Pow(10, (int)Math.Floor(Math.Log10(x)) + 1);

    IEnumerable<int[]> GetChunks(Node node)
    {
        var current = node;
        while (current is not null)
        {
            if (current.Registers.Length != 0)
            {
                yield return current.Registers.ToArray(this);
            }
            current = current.Next;
        }
    }

    Node JoinRecursive(int maxLimit, int decimalOrderMaxLimit, Node root, bool rearrange)
    {
        var node = root;
        while (node is not null)
        {
            var current = node.Registers;
            if (node.Next is not null)
            {
                var follow = node.Next.Registers;
                if (follow.Length == 0)
                {
                    node = node.Next;
                    continue;
                }
                var heightRest = CalculateHeight(node.Next.Next);
                var garbageRest = CalculateGarbage(node.Next.Next);
                var min = CalculateGarbage(current, follow) + garbageRest + decimalOrderMaxLimit * heightRest;
                var prefer = node;
                foreach (var (trimLeft, joinRight) in CombineWithLowerGarbageThanSource(current, follow))
                {
                    if (trimLeft.Length != 0 && ExcessLimit(maxLimit, joinRight, out var taken, out var rest))
                    {
                        if (rearrange)
                        {
                            continue;
                        }
                        var next = JoinRecursive(maxLimit, decimalOrderMaxLimit, CreateNodeWithoutEmptyRegisters([], rest, node.Next.Next), true);
                        if (CalculateHeight(next) <= CalculateHeight(node.Next.Next))
                        {
                            var garbage = CalculateGarbage(trimLeft, taken) + CalculateGarbage(next) + decimalOrderMaxLimit * CalculateHeight(next);
                            if (garbage < min)
                            {
                                min = garbage;
                                prefer = CreateNodeWithoutEmptyRegisters(trimLeft, taken, next);
                            }
                        }
                    }
                    if (!ExcessLimit(maxLimit, joinRight))
                    {
                        var garbage = CalculateGarbage(trimLeft, joinRight) + garbageRest + decimalOrderMaxLimit * (heightRest - (trimLeft.Length == 0 ? 1 : 0));
                        if (garbage < min)
                        {
                            min = garbage;
                            prefer = CreateNodeWithoutEmptyRegisters(trimLeft, joinRight, node.Next.Next);
                        }
                    }
                }
                node.Registers = prefer.Registers;
                node.Next = prefer.Next;
            }
            else
            {
                return root;
            }
            node = node.Next;
        }
        return root;
    }

    Node CreateNodeWithoutEmptyRegisters(RegisterSpan left, RegisterSpan right, Node? rest)
    {
        if (left.Length == 0)
        {
            return new Node()
            {
                Registers = right,
                Next = rest
            };
        }
        if (right.Length == 0)
        {
            return new Node()
            {
                Registers = left,
                Next = rest
            };
        }

        return new Node()
        {
            Registers = left,
            Next = new Node()
            {
                Registers = right,
                Next = rest
            }
        };
    }

    bool ExcessLimit(int maxLimit, ReadOnlySpan<int> chunk, out RegisterSpan taken, out RegisterSpan rest)
    {
        ArgumentOutOfRangeException.ThrowIfZero(chunk.Length);

        if (ExcessLimit(maxLimit, chunk))
        {
            var index = chunk.Length - 1;
            while (index >= 0)
            {
                if (!ExcessLimit(maxLimit, chunk[..index]))
                {
                    break;
                }
                index--;
            }
            rest = chunk[index..].ToArray();
            taken = chunk[..index].ToArray();
            return true;
        }
        rest = RegisterSpan.From(0, 0);
        taken = chunk.ToArray();
        return false;
    }

    bool ExcessLimit(int maxLimit, RegisterSpan chunk) => chunk[^1] - chunk[0] + 1 > maxLimit;

    int CalculateGarbage(RegisterSpan chunk1, RegisterSpan chunk2) => chunk1.Length == 0 ? CalculateGarbage(chunk2) : CalculateGarbage(chunk1) + CalculateGarbage(chunk2);

    int CalculateHeight(Node? node)
    {
        var height = 0;
        var current = node;
        while (current is not null)
        {
            if (current.Registers.Length != 0)
            {
                height++;
            }
            current = current.Next;
        }
        return height;
    }

    int CalculateGarbage(Node? node)
    {
        var garbage = 0;
        var current = node;
        while (current is not null)
        {
            garbage += CalculateGarbage(current.Registers);
            current = current.Next;
        }
        return garbage;
    }

    int CalculateGarbage(RegisterSpan chunk)
    {
        ArgumentOutOfRangeException.ThrowIfZero(chunk.Length);

        var garbage = 0;
        var index = 1;
        while (index < chunk.Length)
        {
            garbage += chunk[this, index] - chunk[this, index - 1] - 1;
            index++;
        }
        return garbage;
    }

    (int[] TrimLeft, int[] JoinRight)[] CombineWithLowerGarbageThanSource(RegisterSpan chunk1, RegisterSpan chunk2)
    {
        List<(int[] TrimLeft, int[] JoinRight)> res = [];
        var min = CalculateGarbage(chunk1, chunk2);
        var concat = chunk1.Concat(chunk2);
        for (var splitPoint = chunk1.Length - 1; splitPoint >= 0; splitPoint--)
        {
            var trimLeft = concat[..splitPoint];
            var joinRight = concat[splitPoint..];
            var garbage = CalculateGarbage(trimLeft, joinRight);
            if (garbage < min || trimLeft.Length == 0)
            {
                min = garbage;
                res.Add((trimLeft.ToArray(), joinRight.ToArray()));
            }
        }
        return res.ToArray();
    }

    Node Chunk()
    {
        var root = new Node();
        var index = 0;
        var previous = registers[0];
        var chunkStart = 0;
        var currentLimit = 1;
        var node = root;
        while (index < registersArray.Length)
        {
            var current = registers[index];
            var distance = current - previous;
            currentLimit += distance;
            if (currentLimit > maxLimit)
            {
                node.Next = new Node()
                {
                    Registers = RegisterSpan.From(chunkStart..index)
                };
                node = node.Next;
                currentLimit = 1;
                chunkStart = index;
            }
            previous = current;
            index++;
        }
        if (currentLimit != 0)
        {
            node.Next = new Node()
            {
                Registers = RegisterSpan.From(chunkStart..index)
            };
        }
        return root;
    }

    public void Dispose() => registersHandle.Free();

    class Node
    {
        [AllowNull] public RegisterSpan Registers;
        public Node? Next;
    }

    class RegisterSpan
    {
        public RegisterSpan(int offset, int length)
        {
            Offset = offset;
            Length = length;
        }

        public int Offset;
        public int Length;

        public int End => Offset + Length;

        public static RegisterSpan Empty = new RegisterSpan(0, 0);

        public int this[AlgorithmInstance algorithm, Index index] => algorithm.registersArray[Offset + index.Value];
        public RegisterSpan this[Range range] => From(range.Start.Value + Offset, range.End.Value + Offset);

        public int[] ToArray(AlgorithmInstance algorithm)
        {
            var array = new int[Length];
            algorithm.registersArray.CopyTo(array, Offset);

            return array;
        }

        public ConcatedRegisterSpan Concat(RegisterSpan with) => new ConcatedRegisterSpan(this, with);

        public static RegisterSpan From(Range range) => From(range.Start.Value, range.End.Value);
        public static RegisterSpan From(int start, int end) => new(start, end - start);
    }

    class ConcatedRegisterSpan
    {
        public ConcatedRegisterSpan(RegisterSpan span1, RegisterSpan span2)
        {
            Span1 = span1;
            Span2 = span2;
        }

        public RegisterSpan Span1;
        public RegisterSpan Span2;

        public ConcatedRegisterSpan this[Range range]
        {
            get
            {
                var start = range.Start.Value;
                var end = range.End.Value;
                var length = end - start;

                var span1 = Span1;
                var span2 = Span2;

                if (end <= span1.Length)
                {

                }

                return new(span1, span2);
            }            
        }
    }
}
*/
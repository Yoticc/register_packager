using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

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

public class Algorithm
{
    public static int[][] Solve(int maxLimit, int[] registers)
    {
        var root = Chunk(maxLimit, registers).Next;
        ArgumentNullException.ThrowIfNull(root);
        var node = JoinRecursive(maxLimit, GetNumberWithZeros(maxLimit), root, false);
        return GetChunks(node).ToArray();
    }

    static int GetNumberWithZeros(int x) => (int)Math.Pow(10, (int)Math.Floor(Math.Log10(x)) + 1);

    static IEnumerable<int[]> GetChunks(Node node)
    {
        var current = node;
        while (current is not null)
        {
            if (current.Registers.Length != 0)
            {
                yield return current.Registers;
            }
            current = current.Next;
        }
    }

    public class Node
    {
        public int[] Registers { get; set; } = [];
        public Node? Next { get; set; }
    }

    private static Node JoinRecursive(int maxLimit, int decimalOrderMaxLimit, Node root, bool rearrange)
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

    private static Node CreateNodeWithoutEmptyRegisters(int[] left, int[] right, Node? rest)
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

    private static bool ExcessLimit(int maxLimit, ReadOnlySpan<int> chunk, out int[] taken, out int[] rest)
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
        taken = chunk.ToArray();
        return false;
    }

    private static bool ExcessLimit(int maxLimit, ReadOnlySpan<int> chunk) => chunk[^1] - chunk[0] + 1 > maxLimit;

    private static int CalculateGarbage(ReadOnlySpan<int> chunk1, ReadOnlySpan<int> chunk2) => chunk1.Length == 0 ? CalculateGarbage(chunk2) : CalculateGarbage(chunk1) + CalculateGarbage(chunk2);

    private static int CalculateHeight(Node? node)
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
    private static int CalculateGarbage(Node? node)
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
    private static int CalculateGarbage(ReadOnlySpan<int> chunk)
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

    private static (int[] TrimLeft, int[] JoinRight)[] CombineWithLowerGarbageThanSource(ReadOnlySpan<int> chunk1, ReadOnlySpan<int> chunk2)
    {
        List<(int[] TrimLeft, int[] JoinRight)> res = [];
        var min = CalculateGarbage(chunk1, chunk2);
        ReadOnlySpan<int> concat = [.. chunk1, .. chunk2];
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

    private static Node Chunk(int maxLimit, int[] registers)
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
}
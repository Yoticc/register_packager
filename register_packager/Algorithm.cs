﻿namespace register_packager;

public class Algorithm
{
    public static int[][] Solve(int maxLimit, int[] registers)
    {
        var node = JoinRecursive(maxLimit, Chunk(maxLimit, registers).Next!, false);
        return GetChunks(node).ToArray();
    }

    private static IEnumerable<int[]> GetChunks(Node node)
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
    
    private static Node JoinRecursive(int maxLimit, Node root, bool rearrange)
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
                var min = CalculateGarbage(current, follow) + CalculateGarbage(node.Next!.Next!) + GetNumberWithZeros(maxLimit) * CalculateHeight(node.Next!.Next!);
                var prefer = node;
                foreach (var (trimLeft, joinRight) in CombineWithLowerGarbageThanSource(current, follow))
                {
                    if (trimLeft.Length != 0 && ExcessLimit(maxLimit, joinRight, out var taken, out var rest))
                    {
                        if (rearrange)
                        {
                            continue;
                        }
                        var n = CreateNodeWithoutEmptyRegisters([], rest, node.Next!.Next!);
                        var next = JoinRecursive(maxLimit, n, true);
                        if (CalculateHeight(next) <= CalculateHeight(node.Next!.Next!))
                        {
                            var g = CalculateGarbage(trimLeft, taken) + CalculateGarbage(next) + GetNumberWithZeros(maxLimit) * CalculateHeight(next);
                            if (g < min)
                            {
                                min = g;
                                prefer = CreateNodeWithoutEmptyRegisters(trimLeft, taken, next);
                            }   
                        }
                    }
                    if (!ExcessLimit(maxLimit, joinRight))
                    {
                        var g = CalculateGarbage(trimLeft, joinRight) + CalculateGarbage(node.Next!.Next!) + GetNumberWithZeros(maxLimit) * (CalculateHeight(node.Next!.Next!) - (trimLeft.Length == 0 ? 1 : 0));
                        if (g < min)
                        {
                            min = g;
                            prefer = CreateNodeWithoutEmptyRegisters(trimLeft, joinRight, node.Next!.Next!);
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

    private static int GetNumberWithZeros(int x) => (int)Math.Pow(10, (int)Math.Floor(Math.Log10(x)) + 1);
    
    private static Node CreateNodeWithoutEmptyRegisters(int[] left, int[] right, Node otherPart)
    {
        if (left.Length == 0)
        {
            return new Node()
            {
                Registers = right,
                Next = otherPart
            };
        }
        if (right.Length == 0)
        {
            return new Node()
            {
                Registers = left,
                Next = otherPart
            };
        }
        return new Node()
        {
            Registers = left,
            Next = new Node()
            {
                Registers = right,
                Next = otherPart
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

    private static int CalculateHeight(Node node)
    {
        var i = 1;
        var current = node;
        while (current is not null)
        {
            if (current.Registers.Length != 0)
            {
                i++;
            }
            current = current.Next;
        }
        return i;
    }
    private static int CalculateGarbage(Node node)
    {
        var i = 0;
        var current = node;
        while (current is not null)
        {
            i += CalculateGarbage(current.Registers);
            current = current.Next;
        }
        return i;
    }
    private static int CalculateGarbage(ReadOnlySpan<int> chunk)
    {
        ArgumentOutOfRangeException.ThrowIfZero(chunk.Length);
        
        var index = 0;
        var garbage = 0;
        var previous = chunk[0];
        while (index < chunk.Length)
        {
            var current = chunk[index];
            garbage += Math.Max(0, current - previous - 1);
            previous = current;
            index++;
        }
        return garbage;
    }
    
    private static (int[] TrimLeft, int[] JoinRight)[] CombineWithLowerGarbageThanSource(ReadOnlySpan<int> chunk1, ReadOnlySpan<int> chunk2)
    {
        List<(int[] TrimLeft, int[] JoinRight)> res = [];
        var min = -1;
        var maxGarbage = CalculateGarbage(chunk1, chunk2);
        ReadOnlySpan<int> concat = [..chunk1, ..chunk2];
        for (var splitPoint = chunk1.Length - 1; splitPoint >= 0; splitPoint--)
        {
            var trimLeft = concat[..splitPoint];
            var joinRight = concat[splitPoint..];
            var garbage = CalculateGarbage(trimLeft, joinRight);
            if (garbage < maxGarbage || trimLeft.Length == 0)
            {
                if (min == -1)
                {
                    min = garbage;
                    res.Add((trimLeft.ToArray(), joinRight.ToArray()));
                }
                if (garbage < min)
                {
                    min = garbage;
                    res.Add((trimLeft.ToArray(), joinRight.ToArray()));
                }
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
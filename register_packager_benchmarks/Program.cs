using BenchmarkDotNet.Running;
using register_packager_benchmarks;

//new Benchmarks().On16394RegistersWithMax256();
BenchmarkRunner.Run<Benchmarks>();
Console.ReadLine();
using BenchmarkDotNet.Running;
using register_packager;
using register_packager_benchmarks;

//new Benchmarks().Reimplemented();
BenchmarkRunner.Run<Benchmarks>();
Console.ReadLine();
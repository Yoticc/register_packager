﻿using BenchmarkDotNet.Running;
using register_packager_benchmarks;

//Console.Write(new Benchmarks().Eq());
//new Benchmarks().Reimplemented();
BenchmarkRunner.Run<Benchmarks>();
Console.ReadLine();
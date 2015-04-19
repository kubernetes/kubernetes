/**
 * Benchmark runner dependencies
 */

var colors = require('colors')
  , path = require('path');

/**
 * Find all the benchmarks
 */

var benchmarks_files = process.env.BENCHMARKS.split(' ')
  , all = [].concat(benchmarks_files)
  , first = all.shift()
  , benchmarks = {};

// find the benchmarks and load them all in our obj
benchmarks_files.forEach(function (file) {
  benchmarks[file] = require(path.join(__dirname, '..', file));
});

// setup the complete listeners
benchmarks_files.forEach(function (file) {
  var benchmark = benchmarks[file]
    , next_file = all.shift()
    , next = benchmarks[next_file];

  /**
   * Generate a oncomplete function for the tests, either we are done or we
   * have more benchmarks to process.
   */

   function complete () {
      if (!next) {
        console.log(
          '\n\nBenchmark completed in'.grey
        , (Date.now() - start).toString().green + ' ms'.grey
        );
      } else {
        console.log('\nStarting benchmark '.grey + next_file.yellow);
        next.run();
      }
   }

   // attach the listener
   benchmark.on('complete', complete);
});

/**
 * Start the benchmark
 */

var start = Date.now();
console.log('Starting benchmark '.grey + first.yellow);
benchmarks[first].run();

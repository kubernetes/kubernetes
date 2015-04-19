// This is to be used by "module.js" (and "module.coffee") example(s).
// There should NOT be a "universe.coffee" as only 1 of the 2 would
//  ever be loaded unless the file extension was specified.

exports.answer = 42;

exports.start = function () {
    console.log('Starting the universe....');
}


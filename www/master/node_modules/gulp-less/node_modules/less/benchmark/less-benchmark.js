var path = require('path'),
    fs = require('fs');

var less = require('../lib/less');
var file = path.join(__dirname, 'benchmark.less');

if (process.argv[2]) { file = path.join(process.cwd(), process.argv[2]) }

fs.readFile(file, 'utf8', function (e, data) {
    var tree, css, start, end, total;

    console.log("Benchmarking...\n", path.basename(file) + " (" +
             parseInt(data.length / 1024) + " KB)", "");

    start = new(Date);

    new(less.Parser)({ optimization: 2 }).parse(data, function (err, tree) {
        end = new Date();

        total = end - start;

        console.log("Parsing: " +
                 total + " ms (" +
                 Number(1000 / total * data.length / 1024) + " KB\/s)");

        start = new Date();
        css = tree.toCSS();
        end = new Date();

        console.log("Generation: " + (end - start) + " ms (" +
                 parseInt(1000 / (end - start) *
                 data.length / 1024) + " KB\/s)");

        total += end - start;

        console.log("Total: " + total + "ms (" +
            Number(1000 / total * data.length / 1024) + " KB/s)");

        if (err) {
            less.writeError(err);
            process.exit(3);
        }
    });
});


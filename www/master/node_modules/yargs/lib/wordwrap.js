// Simplified version of https://github.com/substack/node-wordwrap

'use strict';

module.exports = function (start, stop) {

    if (!stop) {
        stop = start;
        start = 0;
    }

    var re = /(\S+\s+)/;

    return function (text) {
        var chunks = text.toString().split(re);

        return chunks.reduce(function (lines, rawChunk) {
            if (rawChunk === '') return lines;

            var chunk = rawChunk.replace(/\t/g, '    ');

            var i = lines.length - 1;
            if (lines[i].length + chunk.length > stop) {
                lines[i] = lines[i].replace(/\s+$/, '');

                chunk.split(/\n/).forEach(function (c) {
                    lines.push(
                        new Array(start + 1).join(' ')
                            + c.replace(/^\s+/, '')
                    );
                });
            }
            else if (chunk.match(/\n/)) {
                var xs = chunk.split(/\n/);
                lines[i] += xs.shift();
                xs.forEach(function (c) {
                    lines.push(
                        new Array(start + 1).join(' ')
                            + c.replace(/^\s+/, '')
                    );
                });
            }
            else {
                lines[i] += chunk;
            }

            return lines;
        }, [ new Array(start + 1).join(' ') ]).join('\n');
    };
};
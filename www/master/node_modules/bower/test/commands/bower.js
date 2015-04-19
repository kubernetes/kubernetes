var expect = require('expect.js');
var helpers = require('../helpers');

describe('bower', function () {

    var oldStdout;
    var text;

    before(function() {
        oldStdout = process.stdout.write;
        text = '';

        process.stdout.write = function(args) {
            text += args;
        };
    });

    it('runs bower installation', function (done) {
        helpers.require('bin/bower');

        setTimeout(function() {
            done();
        }, 250);
    });

    after(function() {
        process.stdout.write = oldStdout;
        expect(text).to.contain('Usage:');
        expect(text).to.contain('Commands:');
    });
});

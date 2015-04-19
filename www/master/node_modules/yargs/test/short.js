var should = require('chai').should(),
    yargs = require('../');

describe('short options', function () {

    it ('should set n to the numeric value 123', function () {
        var argv = yargs.parse([ '-n123' ]);
        should.exist(argv);
        argv.should.have.property('n', 123);
    });

    it ('should set option "1" to true, option "2" to true, and option "3" to numeric value 456', function () {
        var argv = yargs.parse([ '-123', '456' ]);
        should.exist(argv);
        argv.should.have.property('1', true);
        argv.should.have.property('2', true);
        argv.should.have.property('3', 456);
    });

});

var should = require('chai').should(),
    yargs = require('../');

describe('whitespace', function () {

    it('should be whitespace', function () {
        var argv = yargs.parse([ '-x', '\t' ]);
        should.exist(argv);
        argv.should.have.property('x', '\t');
    });

});

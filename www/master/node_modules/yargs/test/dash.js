var should = require('chai').should(),
    yargs = require('../index');

describe('-', function () {

    it('should set - as value of n', function () {
        var argv = yargs.parse(['-n', '-']);
        argv.should.have.property('n', '-');
        argv.should.have.property('_').with.length(0);
    });

    it('should set - as a non-hyphenated value', function () {
        var argv = yargs.parse(['-']);
        argv.should.have.property('_').and.deep.equal(['-']);
    });

    it('should set - as a value of f', function () {
        var argv = yargs.parse(['-f-']);
        argv.should.have.property('f', '-');
        argv.should.have.property('_').with.length(0);
    });

    it('should set b to true and set - as a non-hyphenated value when b is set as a boolean', function () {
        var argv = yargs(['-b', '-']).boolean('b').argv;
        argv.should.have.property('b', true);
        argv.should.have.property('_').and.deep.equal(['-']);
    });

    it('should set - as the value of s when s is set as a string', function () {
        var argv = yargs([ '-s', '-' ]).string('s').argv;
        argv.should.have.property('s', '-');
        argv.should.have.property('_').with.length(0);
    });

});

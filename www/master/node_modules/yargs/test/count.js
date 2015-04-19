var should = require('chai').should()
    yargs = require('../index');

describe('count', function () {

    it('should count the number of times a boolean is present', function () {
        var parsed;

        parsed = yargs(['-x']).count('verbose').argv;
        parsed.verbose.should.equal(0);

        parsed = yargs(['--verbose']).count('verbose').argv;
        parsed.verbose.should.equal(1);

        parsed = yargs(['--verbose', '--verbose']).count('verbose').argv;
        parsed.verbose.should.equal(2);

        parsed = yargs(['-vvv']).alias('v', 'verbose').count('verbose').argv;
        parsed.verbose.should.equal(3);

        parsed = yargs(['--verbose', '--verbose', '-v', '--verbose']).count('verbose').alias('v', 'verbose').argv;
        parsed.verbose.should.equal(4);

        parsed = yargs(['--verbose', '--verbose', '-v', '-vv']).count('verbose').alias('v', 'verbose').argv;
        parsed.verbose.should.equal(5);
    });

});

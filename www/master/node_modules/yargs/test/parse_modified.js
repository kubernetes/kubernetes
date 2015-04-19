var should = require('chai').should(),
    yargs = require('../');

describe('parse', function () {

    describe('boolean modifier function', function () {

        it('should prevent yargs from sucking in the next option as the value of the first option', function () {

            // Arrange & Act
            var result = yargs().boolean('b').parse([ '-b', '123' ]);

            // Assert
            result.should.have.property('b').that.is.a('boolean').and.is.true;
            result.should.have.property('_').and.deep.equal([123]);

        });

    });

});

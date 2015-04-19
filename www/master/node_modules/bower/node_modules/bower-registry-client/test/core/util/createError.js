var createError = require('../../../lib/util/createError');
var expect = require('expect.js');

describe('createError', function () {

    beforeEach(function () {
        this.err = createError('message', 500);
    });

    describe('requiring the createError module', function () {

        it('should expose a createError method', function () {
            expect(typeof createError === 'function').to.be.ok;
        });

    });

    describe('invoking createError', function () {

        it('should return a new Error Object', function () {
            expect(typeof createError() === 'object').to.be.ok;
        });

        it('should return an Error with message', function () {
            expect(this.err.message).to.eql('message');
        });

        it('should return an Error with code', function () {
            expect(this.err.code).to.eql(500);
        });

    });


});

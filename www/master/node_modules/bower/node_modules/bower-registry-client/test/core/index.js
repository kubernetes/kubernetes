var index = require('../../lib/index');
var expect = require('expect.js');

describe('index module', function () {
    describe('requiring the index module', function () {
        it('should expose a lookup method', function () {
            expect(index.lookup).to.be.ok;
        });

        it('should expose a list method', function () {
            expect(index.list).to.be.ok;
        });

        it('should expose a register method', function () {
            expect(index.register).to.be.ok;
        });

        it('should expose a search method', function () {
            expect(index.search).to.be.ok;
        });
    });
});

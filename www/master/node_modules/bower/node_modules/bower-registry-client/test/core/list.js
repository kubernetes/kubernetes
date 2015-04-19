var list = require('../../lib/list');
var expect = require('expect.js');

describe('list module', function () {
    describe('requiring the list module', function () {
        it('should expose a list method', function () {
            expect(typeof list === 'function').to.be.ok;
        });

        it('should expose a initCache method', function () {
            expect(list.initCache).to.be.ok;
            expect(typeof list.initCache === 'function').to.be.ok;
        });

        it('should expose a clearCache method', function () {
            expect(list.clearCache).to.be.ok;
            expect(typeof list.clearCache === 'function').to.be.ok;
        });

        it('should expose a resetCache method', function () {
            expect(list.resetCache).to.be.ok;
            expect(typeof list.resetCache === 'function').to.be.ok;
        });
    });
});

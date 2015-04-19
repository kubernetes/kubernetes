var search = require('../../lib/search');
var expect = require('expect.js');

describe('search module', function () {
    describe('requiring the search module', function () {
        it('should expose a search method', function () {
            expect(typeof search === 'function').to.be.ok;
        });

        it('should expose a initCache method', function () {
            expect(search.initCache).to.be.ok;
            expect(typeof search.initCache === 'function').to.be.ok;
        });

        it('should expose a clearCache method', function () {
            expect(search.clearCache).to.be.ok;
            expect(typeof search.clearCache === 'function').to.be.ok;
        });

        it('should expose a resetCache method', function () {
            expect(search.resetCache).to.be.ok;
            expect(typeof search.resetCache === 'function').to.be.ok;
        });
    });
});

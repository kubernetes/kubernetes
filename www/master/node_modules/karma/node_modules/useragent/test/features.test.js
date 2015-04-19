describe('useragent/features', function () {
  'use strict';

  var useragent = require('../')
    , features = require('../features')
    , ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_1) AppleWebKit/535.2 (KHTML, like Gecko) Chrome/15.0.874.24 Safari/535.2";

  describe('#satisfies', function () {
    it('should satisfy that range selector', function () {
      var agent = useragent.parse(ua);

      agent.satisfies('15.x || >=19.5.0 || 25.0.0 - 17.2.3').should.be_true;
      agent.satisfies('>16.12.0').should.be_false;
    });
  });
});

describe('useragent', function () {
  'use strict';

  var useragent = require('../')
    , ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_1) AppleWebKit/535.2 (KHTML, like Gecko) Chrome/15.0.874.24 Safari/535.2";

  it('should expose the current version number', function () {
    useragent.version.should.match(/^\d+\.\d+\.\d+$/);
  });

  it('should expose the Agent interface', function () {
    useragent.Agent.should.be.a.Function;
  });

  it('should expose the OperatingSystem interface', function () {
    useragent.OperatingSystem.should.be.a.Function;
  });

  it('should expose the Device interface', function () {
    useragent.Device.should.be.a.Function;
  });

  it('should expose the dictionary lookup', function () {
    useragent.lookup.should.be.a.Function;
  });

  it('should expose the parser', function () {
    useragent.parse.should.be.a.Function;
  });

  it('should expose the useragent tester', function () {
    useragent.is.should.be.a.Function;
  });

  describe('#parse', function () {
    it('correctly transforms everything to the correct instances', function () {
      var agent = useragent.parse(ua);

      agent.should.be.an.instanceOf(useragent.Agent);
      agent.os.should.be.an.instanceOf(useragent.OperatingSystem);
      agent.device.should.be.an.instanceOf(useragent.Device);
    });

    it('correctly parsers the operating system', function () {
      var os = useragent.parse(ua).os;

      os.toString().should.equal('Mac OS X 10.7.1');
      os.toVersion().should.equal('10.7.1');
      JSON.stringify(os).should.equal('{"family":"Mac OS X","major":"10","minor":"7","patch":"1"}');

      os.major.should.equal('10');
      os.minor.should.equal('7');
      os.patch.should.equal('1');
    });

    it('should not throw errors when no useragent is given', function () {
      var agent = useragent.parse();

      agent.family.should.equal('Other');
      agent.major.should.equal('0');
      agent.minor.should.equal('0');
      agent.patch.should.equal('0');

      agent.os.toString().should.equal('Other');
      agent.toVersion().should.equal('0.0.0');
      agent.toString().should.equal('Other 0.0.0 / Other');
      agent.toAgent().should.equal('Other 0.0.0');
      JSON.stringify(agent).should.equal('{"family":"Other","major":"0","minor":"0","patch":"0","device":{"family":"Other"},"os":{"family":"Other"}}');
    });

    it('should not throw errors on empty strings and default to unkown', function () {
      var agent = useragent.parse('');

      agent.family.should.equal('Other');
      agent.major.should.equal('0');
      agent.minor.should.equal('0');
      agent.patch.should.equal('0');

      agent.os.toString().should.equal('Other');
      agent.toVersion().should.equal('0.0.0');
      agent.toString().should.equal('Other 0.0.0 / Other');
      agent.toAgent().should.equal('Other 0.0.0');
      JSON.stringify(agent).should.equal('{"family":"Other","major":"0","minor":"0","patch":"0","device":{"family":"Other"},"os":{"family":"Other"}}');
    });

    it('should correctly parse chromes user agent', function () {
      var agent = useragent.parse(ua);

      agent.family.should.equal('Chrome');
      agent.major.should.equal('15');
      agent.minor.should.equal('0');
      agent.patch.should.equal('874');

      agent.os.toString().should.equal('Mac OS X 10.7.1');
      agent.toVersion().should.equal('15.0.874');
      agent.toString().should.equal('Chrome 15.0.874 / Mac OS X 10.7.1');
      agent.toAgent().should.equal('Chrome 15.0.874');
      JSON.stringify(agent).should.equal('{"family":"Chrome","major":"15","minor":"0","patch":"874","device":{"family":"Other"},"os":{"family":"Mac OS X","major":"10","minor":"7","patch":"1"}}');
    });

    it('correctly parses iOS8', function () {
      var agent = useragent.parse('Mozilla/5.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12A365 Safari/600.1.4');

      agent.os.family.should.equal('iOS');
      agent.os.major.should.equal('8');
    });
  });

  describe('#fromJSON', function () {
    it('should re-generate the Agent instance', function () {
      var agent = useragent.parse(ua)
        , string = JSON.stringify(agent)
        , agent2 = useragent.fromJSON(string);

      agent2.family.should.equal(agent.family);
      agent2.major.should.equal(agent.major);
      agent2.minor.should.equal(agent.minor);
      agent2.patch.should.equal(agent.patch);

      agent2.device.family.should.equal(agent.device.family);

      agent2.os.family.should.equal(agent.os.family);
      agent2.os.major.should.equal(agent.os.major);
      agent2.os.minor.should.equal(agent.os.minor);
      agent2.os.patch.should.equal(agent.os.patch);
    });

    it('should also work with legacy JSON', function () {
      var agent = useragent.fromJSON('{"family":"Chrome","major":"15","minor":"0","patch":"874","os":"Mac OS X"}');

      agent.family.should.equal('Chrome');
      agent.major.should.equal('15');
      agent.minor.should.equal('0');
      agent.patch.should.equal('874');

      agent.device.family.should.equal('Other');

      agent.os.family.should.equal('Mac OS X');
    });
  });

  describe('#is', function () {
    var chrome = ua
      , firefox = 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:8.0) Gecko/20100101 Firefox/8.0'
      , ie = 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0; yie8)'
      , ie11 = 'Mozilla/5.0 (Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko'
      , opera = 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; de) Opera 11.51'
      , safari = 'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_7; da-dk) AppleWebKit/533.21.1 (KHTML, like Gecko) Version/5.0.5 Safari/533.21.1'
      , ipod = 'Mozilla/5.0 (iPod; U; CPU iPhone OS 4_3_3 like Mac OS X; ja-jp) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5'
      , android = 'Mozilla/5.0 (Linux; U; Android 2.3.6; en-us; Nexus S Build/GRK39F) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1';

    it('should not throw errors when called without arguments', function () {
      useragent.is();
      useragent.is('');
    });

    it('should correctly detect google chrome', function () {
      useragent.is(chrome).chrome.should.equal(true);
      useragent.is(chrome).webkit.should.equal(true);
      useragent.is(chrome).safari.should.equal(false);
      useragent.is(chrome).firefox.should.equal(false);
      useragent.is(chrome).mozilla.should.equal(false);
      useragent.is(chrome).ie.should.equal(false);
      useragent.is(chrome).opera.should.equal(false);
      useragent.is(chrome).mobile_safari.should.equal(false);
      useragent.is(chrome).android.should.equal(false);
    });

    it('should correctly detect firefox', function () {
      useragent.is(firefox).chrome.should.equal(false);
      useragent.is(firefox).webkit.should.equal(false);
      useragent.is(firefox).safari.should.equal(false);
      useragent.is(firefox).firefox.should.equal(true);
      useragent.is(firefox).mozilla.should.equal(true);
      useragent.is(firefox).ie.should.equal(false);
      useragent.is(firefox).opera.should.equal(false);
      useragent.is(firefox).mobile_safari.should.equal(false);
      useragent.is(firefox).android.should.equal(false);
    });

    it('should correctly detect internet explorer', function () {
      useragent.is(ie).chrome.should.equal(false);
      useragent.is(ie).webkit.should.equal(false);
      useragent.is(ie).safari.should.equal(false);
      useragent.is(ie).firefox.should.equal(false);
      useragent.is(ie).mozilla.should.equal(false);
      useragent.is(ie).ie.should.equal(true);
      useragent.is(ie).opera.should.equal(false);
      useragent.is(ie).mobile_safari.should.equal(false);
      useragent.is(ie).android.should.equal(false);

      useragent.is(ie11).chrome.should.equal(false);
      useragent.is(ie11).webkit.should.equal(false);
      useragent.is(ie11).safari.should.equal(false);
      useragent.is(ie11).firefox.should.equal(false);
      useragent.is(ie11).mozilla.should.equal(false);
      useragent.is(ie11).ie.should.equal(true);
      useragent.is(ie11).opera.should.equal(false);
      useragent.is(ie11).mobile_safari.should.equal(false);
      useragent.is(ie11).android.should.equal(false);
    });

    it('should correctly detect opera', function () {
      useragent.is(opera).chrome.should.equal(false);
      useragent.is(opera).webkit.should.equal(false);
      useragent.is(opera).safari.should.equal(false);
      useragent.is(opera).firefox.should.equal(false);
      useragent.is(opera).mozilla.should.equal(false);
      useragent.is(opera).ie.should.equal(false);
      useragent.is(opera).opera.should.equal(true);
      useragent.is(opera).mobile_safari.should.equal(false);
      useragent.is(opera).android.should.equal(false);
    });

    it('should correctly detect safari', function () {
      useragent.is(safari).chrome.should.equal(false);
      useragent.is(safari).webkit.should.equal(true);
      useragent.is(safari).safari.should.equal(true);
      useragent.is(safari).firefox.should.equal(false);
      useragent.is(safari).mozilla.should.equal(false);
      useragent.is(safari).ie.should.equal(false);
      useragent.is(safari).opera.should.equal(false);
      useragent.is(safari).mobile_safari.should.equal(false);
      useragent.is(safari).android.should.equal(false);
    });

    it('should correctly detect safari-mobile', function () {
      useragent.is(ipod).chrome.should.equal(false);
      useragent.is(ipod).webkit.should.equal(true);
      useragent.is(ipod).safari.should.equal(true);
      useragent.is(ipod).firefox.should.equal(false);
      useragent.is(ipod).mozilla.should.equal(false);
      useragent.is(ipod).ie.should.equal(false);
      useragent.is(ipod).opera.should.equal(false);
      useragent.is(ipod).mobile_safari.should.equal(true);
      useragent.is(ipod).android.should.equal(false);
    });

    it('should correctly detect android', function () {
      useragent.is(android).chrome.should.equal(false);
      useragent.is(android).webkit.should.equal(true);
      useragent.is(android).safari.should.equal(true);
      useragent.is(android).firefox.should.equal(false);
      useragent.is(android).mozilla.should.equal(false);
      useragent.is(android).ie.should.equal(false);
      useragent.is(android).opera.should.equal(false);
      useragent.is(android).mobile_safari.should.equal(true);
      useragent.is(android).android.should.equal(true);
    });
  });
});

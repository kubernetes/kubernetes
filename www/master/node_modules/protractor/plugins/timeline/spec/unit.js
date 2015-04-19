var fs = require('fs');
var path = require('path');
var TimelinePlugin = require('../index.js').TimelinePlugin;

describe('timeline plugin', function() {
  it('should parse an example selenium standalone log', function() {
    var timeline = TimelinePlugin.parseTextLog(
        fs.readFileSync(path.join(__dirname, 'standalonelog.txt')).toString());

    expect(timeline.length).toEqual(23);
    expect(timeline[0].command).toEqual('new session');
    expect(timeline[0].duration).toEqual(3424);
  });

  it('should parse an example chromedriver log', function() {
    var timeline = TimelinePlugin.parseTextLog(
        fs.readFileSync(path.join(__dirname, 'chromelog.txt')).toString());

    expect(timeline.length).toEqual(24);
    expect(timeline[0].command).toEqual('InitSession');
    expect(timeline[0].duration).toEqual(1781);
  });

  it('should parse example selenium client logs', function() {
    var clientLog = [{
      level: { value: 800, name: 'INFO' },
      message: 'org.openqa.selenium.remote.server.DriverServlet org.openqa.selenium.remote.server.rest.ResultConfig.handle Executing: [click: 1 [[SafariDriver: safari on MAC (null)] -> css selector: [ng-click="slowHttp()"]]] at URL: /session/b057191b-6328-4c7b-bdd2-170ba79280af/element/1/click)',
      timestamp: 1419031697443,
      type: ''
    }, { level: { value: 800, name: 'INFO' },
      message: 'org.openqa.selenium.remote.server.DriverServlet org.openqa.selenium.remote.server.rest.ResultConfig.handle Done: /session/b057191b-6328-4c7b-bdd2-170ba79280af/element/1/click',
      timestamp: 1419031697470,
      type: ''
    }, { level: { value: 800, name: 'INFO' },
      message: 'org.openqa.selenium.remote.server.DriverServlet org.openqa.selenium.remote.server.rest.ResultConfig.handle Executing: [execute async script: try { return (function (rootSelector, callback) {\n  var el = document.querySelector(rootSelector);\n\n  try {\n    if (!window.angular) {\n      throw new Error(\'angular could not be found on the window\');\n    }\n    if (angular.getTestability) {\n      angular.getTestability(el).whenStable(callback);\n    } else {\n      if (!angular.element(el).injector()) {\n        throw new Error(\'root element (\' + rootSelector + \') has no injector.\' +\n           \' this may mean it is not inside ng-app.\');\n      }\n      angular.element(el).injector().get(\'$browser\').\n          notifyWhenNoOutstandingRequests(callback);\n    }\n  } catch (err) {\n    callback(err.message);\n  }\n}).apply(this, arguments); }\ncatch(e) { throw (e instanceof Error) ? e : new Error(e); }, [body]] at URL: /session/b057191b-6328-4c7b-bdd2-170ba79280af/execute_async)',
      timestamp: 1419031697672,
      type: ''
    }, { level: { value: 800, name: 'INFO' },
      message: 'org.openqa.selenium.remote.server.DriverServlet org.openqa.selenium.remote.server.rest.ResultConfig.handle Done: /session/b057191b-6328-4c7b-bdd2-170ba79280af/execute_async',
      timestamp: 1419031702495,
      type: ''
    }, { level: { value: 800, name: 'INFO' },
      message: 'org.openqa.selenium.remote.server.DriverServlet org.openqa.selenium.remote.server.rest.ResultConfig.handle Executing: [execute script: try { return (function (binding, exactMatch, using, rootSelector) {\n  var root = document.querySelector(rootSelector || \'body\');\n  using = using || document;\n  if (angular.getTestability) {\n    return angular.getTestability(root).\n        findBindings(using, binding, exactMatch);\n  }\n  var bindings = using.getElementsByClassName(\'ng-binding\');\n  var matches = [];\n  for (var i = 0; i < bindings.length; ++i) {\n    var dataBinding = angular.element(bindings[i]).data(\'$binding\');\n    if(dataBinding) {\n      var bindingName = dataBinding.exp || dataBinding[0].exp || dataBinding;\n      if (exactMatch) {\n        var matcher = new RegExp(\'({|\\\\s|^|\\\\|)\' + binding + \'(}|\\\\s|$|\\\\|)\');\n        if (matcher.test(bindingName)) {\n          matches.push(bindings[i]);\n        }\n      } else {\n        if (bindingName.indexOf(binding) != -1) {\n          matches.push(bindings[i]);\n        }\n      }\n      \n    }\n  }\n  return matches; /* Return the whole array for webdriver.findElements. */\n}).apply(this, arguments); }\ncatch(e) { throw (e instanceof Error) ? e : new Error(e); }, [slowHttpStatus, false, null, body]] at URL: /session/b057191b-6328-4c7b-bdd2-170ba79280af/execute)',
      timestamp: 1419031702585,
      type: ''
    }, { level: { value: 800, name: 'INFO' },
      message: 'org.openqa.selenium.remote.server.DriverServlet org.openqa.selenium.remote.server.rest.ResultConfig.handle Done: /session/b057191b-6328-4c7b-bdd2-170ba79280af/execute',
      timestamp: 1419031702608,
      type: ''
    }, { level: { value: 800, name: 'INFO' },
      message: 'org.openqa.selenium.remote.server.DriverServlet org.openqa.selenium.remote.server.rest.ResultConfig.handle Executing: [get text: 0 [org.openqa.selenium.remote.RemoteWebElement@2a868887 -> unknown locator]] at URL: /session/b057191b-6328-4c7b-bdd2-170ba79280af/element/0/text)',
      timestamp: 1419031702795,
      type: ''
    }, { level: { value: 800, name: 'INFO' },
      message: 'org.openqa.selenium.remote.server.DriverServlet org.openqa.selenium.remote.server.rest.ResultConfig.handle Done: /session/b057191b-6328-4c7b-bdd2-170ba79280af/element/0/text',
      timestamp: 1419031702812,
      type: ''
    }];
    var timeline = TimelinePlugin.parseArrayLog(clientLog);

    expect(timeline.length).toEqual(4);
    expect(timeline[0].command).toEqual('click');
    expect(timeline[0].duration).toEqual(1419031697470 - 1419031697443);
  });

  it('should align chrome logs', function() {
    var chromeLog =
        '[1.000][INFO]: COMMAND SetScriptTimeout {\n' +
        '  "ms": 11000\n' +
        '}\n' +
        '[1.000][INFO]: RESPONSE SetScriptTimeout\n' +
        '[4.000][INFO]: COMMAND GetElementText {\n' +
        '  "id": "0.5319263362325728-1"\n' +
        '}\n' +
        '[5.000][INFO]: RESPONSE GetElementText "not started"\n';
    var timeline = TimelinePlugin.parseTextLog(chromeLog, 'chrome logs', 90000);
    expect(timeline.length).toEqual(2);
    expect(timeline[1].source).toEqual('chrome logs');
    expect(timeline[1].start).toEqual(93000);
    expect(timeline[1].end).toEqual(94000);
  });

  it('should align selenium standalone logs', function() {
    var clientLog =
        'INFO: Launching a standalone server\n' +
        '01:40:01.0000 INFO - Executing: [set script timeoutt: 5] at URL: /abc)\n' +
        '01:40:02.0000 INFO - Done: /abc\n' +
        '01:40:04.0000 INFO - Executing: [get text: 0 [elem -> unknown locator]] at URL: /abc)\n' +
        '01:40:05.0000 INFO - Done: /abc\n';

    var timeline = TimelinePlugin.parseTextLog(clientLog, 'standalone log', 90000);
    expect(timeline.length).toEqual(2);
    expect(timeline[1].source).toEqual('standalone log');
    expect(timeline[1].start).toEqual(93000);
    expect(timeline[1].end).toEqual(94000);
  });

  it('should align selenium client logs', function() {
    var clientLog = [{
      level: { value: 800, name: 'INFO' },
      message: 'foo.bar Executing: [set script timeout: 1]',
      timestamp: 111000,
      type: ''
    }, { level: { value: 800, name: 'INFO' },
      message: 'foo.bar Done: [set script timeout: 1]',
      timestamp: 112000,
      type: ''
    }, { level: { value: 800, name: 'INFO' },
      message: 'foo.bar Executing: [get text: 0 [elemid -> unknown locator]]',
      timestamp: 114000,
      type: ''
    }, { level: { value: 800, name: 'INFO' },
      message: 'foo.bar Done: [get text: 0 [elemid -> unknown locator]]',
      timestamp: 115000,
      type: ''
    }];
    var timeline = TimelinePlugin.parseArrayLog(clientLog, 'client log', 90000);

    expect(timeline.length).toEqual(2);
    expect(timeline[1].source).toEqual('client log');
    expect(timeline[1].start).toEqual(93000);
    expect(timeline[1].end).toEqual(94000);
  });
});

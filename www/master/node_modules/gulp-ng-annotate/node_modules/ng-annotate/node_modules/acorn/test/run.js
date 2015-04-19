(function() {
  var driver;

  if (typeof require !== "undefined") {
    driver = require("./driver.js");
    require("./tests.js");
    require("./tests-harmony.js");
  } else {
    driver = window;
  }

  var htmlLog = typeof document === "object" && document.getElementById('log');
  var htmlGroup = htmlLog;

  function group(name) {
    if (htmlGroup) {
      var parentGroup = htmlGroup;
      htmlGroup = document.createElement("ul");
      var item = document.createElement("li");
      item.textContent = name;
      item.appendChild(htmlGroup);
      parentGroup.appendChild(item);
    }
    if (typeof console === "object" && console.group) {
      console.group(name);
    }
  }

  function groupEnd() {
    if (htmlGroup) {
      htmlGroup = htmlGroup.parentElement.parentElement;
    }
    if (typeof console === "object" && console.groupEnd) {
      console.groupEnd(name);
    }
  }

  function log(title, message) {
    if (htmlGroup) {
      var elem = document.createElement("li");
      elem.innerHTML = "<b>" + title + "</b> " + message;
      htmlGroup.appendChild(elem);
    }
    if (typeof console === "object") console.log(title, message);
  }

  var stats, modes = {
    Normal: {
      config: {
        parse: (typeof require === "undefined" ? window.acorn : require("../acorn.js")).parse
      }
    },
    Loose: {
      config: {
        parse: (typeof require === "undefined" ? window.acorn : require("../acorn_loose")).parse_dammit,
        loose: true,
        filter: function (test) {
          var opts = test.options || {};
          if (opts.loose === false) return false;
          return (opts.ecmaVersion || 5) <= 6;
        }
      }
    }
  };

  function report(state, code, message) {
    if (state != "ok") {++stats.failed; log(code, message);}
    ++stats.testsRun;
  }

  group("Errors");

  for (var name in modes) {
    group(name);
    var mode = modes[name];
    stats = mode.stats = {testsRun: 0, failed: 0};
    var t0 = +new Date;
    driver.runTests(mode.config, report);
    mode.stats.duration = +new Date - t0;
    groupEnd();
  }

  groupEnd();

  function outputStats(name, stats) {
    log(name + ":", stats.testsRun + " tests run in " + stats.duration + "ms; " +
      (stats.failed ? stats.failed + " failures." : "all passed."));
  }

  var total = {testsRun: 0, failed: 0, duration: 0};

  group("Stats");

  for (var name in modes) {
    var stats = modes[name].stats;
    outputStats(name + " parser", stats);
    for (var key in stats) total[key] += stats[key];
  }

  outputStats("Total", total);

  groupEnd();

  if (total.failed && typeof process === "object") {
    process.stdout.write("", function() {
      process.exit(1);
    });
  }
})();

/* record log messages for testing */
// var logAllIds = function() {
//   var allTags = document.head.getElementsByTagName('style');
//   var ids = [];
//   for (var tg = 0; tg < allTags.length; tg++) {
//     var tag = allTags[tg];
//     if (tag.id) {
//       console.log(tag.id);
//     }
//   }
// };

var logMessages = [],
  realConsoleLog = console.log;
console.log = function(msg) {
  logMessages.push(msg);
  realConsoleLog.call(console, msg);
};

var testLessEqualsInDocument = function() {
  testLessInDocument(testSheet);
};

var testLessErrorsInDocument = function(isConsole) {
    testLessInDocument(isConsole ? testErrorSheetConsole : testErrorSheet);
};

var testLessInDocument = function(testFunc) {
  var links = document.getElementsByTagName('link'),
    typePattern = /^text\/(x-)?less$/;

  for (var i = 0; i < links.length; i++) {
    if (links[i].rel === 'stylesheet/less' || (links[i].rel.match(/stylesheet/) &&
      (links[i].type.match(typePattern)))) {
      testFunc(links[i]);
    }
  }
};

var testSheet = function(sheet) {
  it(sheet.id + " should match the expected output", function() {
    var lessOutputId = sheet.id.replace("original-", ""),
      expectedOutputId = "expected-" + lessOutputId,
      lessOutputObj,
      lessOutput,
      expectedOutputHref = document.getElementById(expectedOutputId).href,
      expectedOutput = loadFile(expectedOutputHref);

    // Browser spec generates less on the fly, so we need to loose control
    waitsFor(function() {
      lessOutputObj = document.getElementById(lessOutputId);
      // the type condition is necessary because of inline browser tests
      return lessOutputObj !== null && lessOutputObj.type === "text/css";
    }, "generation of " + lessOutputId + "", 700);

    runs(function() {
      lessOutput = lessOutputObj.innerText || lessOutputObj.innerHTML;
    });

    waitsFor(function() {
      return expectedOutput.loaded;
    }, "failed to load expected outout", 10000);

    runs(function() {
      // use sheet to do testing
            expect(expectedOutput.text).toEqual(lessOutput);
    });
  });
};

//TODO: do it cleaner - the same way as in css

function extractId(href) {
  return href.replace(/^[a-z-]+:\/+?[^\/]+/, '') // Remove protocol & domain
  .replace(/^\//, '') // Remove root /
  .replace(/\.[a-zA-Z]+$/, '') // Remove simple extension
  .replace(/[^\.\w-]+/g, '-') // Replace illegal characters
  .replace(/\./g, ':'); // Replace dots with colons(for valid id)
}

var testErrorSheet = function(sheet) {
  it(sheet.id + " should match an error", function() {
    var lessHref = sheet.href,
      id = "less-error-message:" + extractId(lessHref),
      //            id = sheet.id.replace(/^original-less:/, "less-error-message:"),
      errorHref = lessHref.replace(/.less$/, ".txt"),
      errorFile = loadFile(errorHref),
      actualErrorElement,
      actualErrorMsg;

    // Less.js sets 10ms timer in order to add error message on top of page.
    waitsFor(function() {
      actualErrorElement = document.getElementById(id);
      return actualErrorElement !== null;
    }, "error message was not generated", 70);

    runs(function() {
      actualErrorMsg = actualErrorElement.innerText
        .replace(/\n\d+/g, function(lineNo) {
        return lineNo + " ";
      })
        .replace(/\n\s*in /g, " in ")
        .replace("\n\n", "\n");
    });

    waitsFor(function() {
      return errorFile.loaded;
    }, "failed to load expected outout", 10000);

    runs(function() {
      var errorTxt = errorFile.text
        .replace("{path}", "")
        .replace("{pathrel}", "")
        .replace("{pathhref}", "http://localhost:8081/test/less/errors/")
        .replace("{404status}", " (404)");
      expect(errorTxt).toEqual(actualErrorMsg);
      if (errorTxt == actualErrorMsg) {
        actualErrorElement.style.display = "none";
      }
        });
    });
};

var testErrorSheetConsole = function(sheet) {
    it(sheet.id + " should match an error", function() {
        var lessHref =  sheet.href,
            id = sheet.id.replace(/^original-less:/, "less-error-message:"),
            errorHref = lessHref.replace(/.less$/, ".txt"),
            errorFile = loadFile(errorHref),
            actualErrorElement = document.getElementById(id),
            actualErrorMsg = logMessages[logMessages.length - 1];

        describe("the error", function() {
            expect(actualErrorElement).toBe(null);

        });

        /*actualErrorMsg = actualErrorElement.innerText
            .replace(/\n\d+/g, function(lineNo) { return lineNo + " "; })
            .replace(/\n\s*in /g, " in ")
            .replace("\n\n", "\n");*/

        waitsFor(function() {
            return errorFile.loaded;
        }, "failed to load expected outout", 10000);

        runs(function() {
            var errorTxt = errorFile.text
                .replace("{path}", "")
                .replace("{pathrel}", "")
                .replace("{pathhref}", "http://localhost:8081/browser/less/")
                .replace("{404status}", " (404)")
                .trim();
            expect(errorTxt).toEqual(actualErrorMsg);
    });
  });
};

var loadFile = function(href) {
  var request = new XMLHttpRequest(),
    response = {
      loaded: false,
      text: ""
    };
  request.open('GET', href, true);
  request.onload = function(e) {
    response.text = request.response.replace(/\r/g, "");
    response.loaded = true;
  };
  request.send();
  return response;
};

(function() {
  var jasmineEnv = jasmine.getEnv();
  jasmineEnv.updateInterval = 1000;

  var htmlReporter = new jasmine.HtmlReporter();

  jasmineEnv.addReporter(htmlReporter);

  jasmineEnv.specFilter = function(spec) {
    return htmlReporter.specFilter(spec);
  };

  var currentWindowOnload = window.onload;

  window.onload = function() {
    if (currentWindowOnload) {
      currentWindowOnload();
    }
    execJasmine();
  };

  function execJasmine() {
    setTimeout(function() {
      jasmineEnv.execute();
    }, 3000);
  }

})();
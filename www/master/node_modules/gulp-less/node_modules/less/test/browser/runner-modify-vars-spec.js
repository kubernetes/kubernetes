var alreadyRun = false;

describe("less.js modify vars", function() {
  beforeEach(function() {
    // simulating "setUp" or "beforeAll" method
    var lessOutputObj;
    if (alreadyRun)
      return;

    alreadyRun = true;

    // wait until the sheet is compiled first time
    waitsFor(function() {
      lessOutputObj = document.getElementById("less:test-less-simple");
      return lessOutputObj !== null;
    }, "first generation of less:test-less-simple", 7000);

    // modify variables
    runs(function() {
      lessOutputObj.type = "not compiled yet";
      less.modifyVars({
        var1: "green",
        var2: "purple",
        scale: 20
      });
    });

    // wait until variables are modified
    waitsFor(function() {
      lessOutputObj = document.getElementById("less:test-less-simple");
      return lessOutputObj !== null && lessOutputObj.type === "text/css";
    }, "second generation of less:test-less-simple", 7000);

  });

  testLessEqualsInDocument();
  it("Should log only 2 XHR requests", function() {
    var xhrLogMessages = logMessages.filter(function(item) {
      return (/XHR: Getting '/).test(item);
    });
    expect(xhrLogMessages.length).toEqual(2);
  });
});
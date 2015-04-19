var lessTest = require("./less-test"),
    lessTester = lessTest(),
    path = require("path"),
    stylize = require('../lib/less/lessc_helper').stylize;

function getErrorPathReplacementFunction(dir) {
    return function(input) {
        return input.replace(
                "{path}", path.join(process.cwd(), "/test/less/" + dir + "/"))
            .replace("{pathrel}", path.join("test", "less", dir + "/"))
            .replace("{pathhref}", "")
            .replace("{404status}", "")
            .replace(/\r\n/g, '\n');
    };
}

console.log("\n" + stylize("Less", 'underline') + "\n");
lessTester.runTestSet({strictMath: true, relativeUrls: true, silent: true});
lessTester.runTestSet({strictMath: true, strictUnits: true}, "errors/",
    lessTester.testErrors, null, getErrorPathReplacementFunction("errors"));
lessTester.runTestSet({strictMath: true, strictUnits: true, javascriptEnabled: false}, "no-js-errors/",
    lessTester.testErrors, null, getErrorPathReplacementFunction("no-js-errors"));
lessTester.runTestSet({strictMath: true, dumpLineNumbers: 'comments'}, "debug/", null,
    function(name) { return name + '-comments'; });
lessTester.runTestSet({strictMath: true, dumpLineNumbers: 'mediaquery'}, "debug/", null,
    function(name) { return name + '-mediaquery'; });
lessTester.runTestSet({strictMath: true, dumpLineNumbers: 'all'}, "debug/", null,
    function(name) { return name + '-all'; });
lessTester.runTestSet({strictMath: true, relativeUrls: false, rootpath: "folder (1)/"}, "static-urls/");
lessTester.runTestSet({strictMath: true, compress: true}, "compression/");
lessTester.runTestSet({}, "legacy/");
lessTester.runTestSet({strictMath: true, strictUnits: true, sourceMap: true, globalVars: true }, "sourcemaps/",
    lessTester.testSourcemap, null, null, 
    function(filename, type) { 
        if (type === "vars") {
            return path.join('test/less/', filename) + '.json';
        }
        return path.join('test/sourcemaps', filename) + '.json'; 
    });
lessTester.runTestSet({globalVars: true, banner: "/**\n  * Test\n  */\n"}, "globalVars/",
    null, null, null, function(name) { return path.join('test/less/', name) + '.json'; });
lessTester.runTestSet({modifyVars: true}, "modifyVars/",
    null, null, null, function(name) { return path.join('test/less/', name) + '.json'; });
lessTester.runTestSet({urlArgs: '424242'}, "url-args/");
lessTester.testNoOptions();

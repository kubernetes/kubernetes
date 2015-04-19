// run-tests.js
// MIT licensed, see LICENSE file
// Copyright (c) 2013-2015 Olov Lassus <olov.lassus@gmail.com>

"use strict";

var ngAnnotate = require("./ng-annotate-main");
var fs = require("fs");
var os = require("os");
var path = require("path");
var diff = require("diff");
var findLineColumn = require("find-line-column");
var fmt = require("simple-fmt");
var SourceMapConsumer = require("source-map").SourceMapConsumer;
var coffee = require("coffee-script");
var convertSourceMap = require("convert-source-map");

function slurp(filename) {
    return String(fs.readFileSync(filename));
}

function time(str, fn) {
    var t0 = Date.now();
    fn();
    var t1 = Date.now();
    console.log(fmt(str, t1 - t0));
}

function test(correct, got, name) {
    if (got !== correct) {
        var patch = diff.createPatch(name, correct, got);
        process.stderr.write(patch);
        process.exit(-1);
    }
}

var renameOptions = [
    {"from": "$a", "to": "$aRenamed"},
    {"from": "$b", "to": "$bRenamed"},
    {"from": "$c", "to": "$cRenamed"},
    {"from": "$d", "to": "$dRenamed"},
    {"from": "$e", "to": "$eRenamed"},
    {"from": "$f", "to": "$fRenamed"},
    {"from": "$g", "to": "$gRenamed"},
    {"from": "$h", "to": "$hRenamed"},
    {"from": "$i", "to": "$iRenamed"},
];

function testSourcemap(original, got, sourcemap) {
    var smc = new SourceMapConsumer(sourcemap);

    function stringRegExp(commentText) {
        return new RegExp("\"" + commentText + "\"");
    }

    function functionRegExp(functionName) {
        return new RegExp("(function)?\\(" + functionName + "_param1, " + functionName + "_param2\\)")
    }

    function testMapping(needle) {
        var gotResult = needle.exec(got);
        if (gotResult == null) {
            process.stderr.write(fmt("Couldn't find {0} in output source", needle));
            process.exit(-1);
        }

        var expectedResult = needle.exec(original);
        if (expectedResult == null) {
            process.stderr.write(fmt("Couldn't find {0} in expected source", needle));
            process.exit(-1);
        }

        var gotPosition = findLineColumn(got, gotResult.index);
        var originalPosition = smc.originalPositionFor({ line: gotPosition.line, column: gotPosition.col });
        var expectedPosition = findLineColumn(original, expectedResult.index);

        if (originalPosition.line !== expectedPosition.line || originalPosition.column !== expectedPosition.col) {
            process.stderr.write(fmt("Sourcemap mapping error for {0}. Expected: ({1},{2}) => ({3},{4}). Got: ({5},{6}) => ({3},{4}).",
                needle,
                expectedPosition.line, expectedPosition.col,
                gotPosition.line, gotPosition.col,
                originalPosition.line, originalPosition.column));
            process.exit(-1);
        }
    }

    testMapping(stringRegExp("before"));
    for (var i = 1; i <= 4; i++) {
        testMapping(functionRegExp("ctrl" + i));
        testMapping(stringRegExp("ctrl" + i + " body"));
    }
    testMapping(stringRegExp("after"));
}

function run(ngAnnotate) {
    var original = slurp("tests/original.js");

    console.log("testing adding annotations");
    var annotated = ngAnnotate(original, {add: true}).src;
    test(slurp("tests/with_annotations.js"), annotated, "with_annotations.js");

    var rename = slurp("tests/rename.js");

    console.log("testing adding annotations and renaming");
    var annotatedRenamed = ngAnnotate(rename, {
        add: true,
        rename: renameOptions,
    }).src;
    test(slurp("tests/rename.annotated.js"), annotatedRenamed, "rename.annotated.js");

    console.log("testing removing annotations");
    test(original, ngAnnotate(annotated, {remove: true}).src, "original.js");

    console.log("testing adding annotations twice");
    test(annotated, ngAnnotate(annotated, {add: true}).src, "with_annotations.js");

    console.log("testing rebuilding annotations");
    test(annotated, ngAnnotate(annotated, {add: true, remove: true}).src, "with_annotations.js");

    console.log("testing adding existing $inject annotations (no change)");
    test(slurp("tests/has_inject.js"), ngAnnotate(slurp("tests/has_inject.js"), {add: true}).src);

    console.log("testing removing existing $inject annotations");
    test(slurp("tests/has_inject_removed.js"), ngAnnotate(slurp("tests/has_inject.js"), {remove: true}).src);

    console.log("testing sourcemaps");
    var originalSourcemaps = slurp("tests/sourcemaps.coffee");
    var compiledSourcemaps = coffee.compile(originalSourcemaps, { sourceFiles: ["sourcemaps.coffee"], generatedFile: "sourcemaps.js", sourceMap: true });
    var annotatedSourcemaps = ngAnnotate(compiledSourcemaps.js, {remove: true, add: true, sourcemap: { sourceRoot: "/source/root/dir" }});
    test(slurp("tests/sourcemaps.annotated.js"), annotatedSourcemaps.src, "sourcemaps.annotated.js");
    testSourcemap(compiledSourcemaps.js, annotatedSourcemaps.src, annotatedSourcemaps.map, "sourcemaps.annotated.js.map");

    console.log("testing sourcemap combination");
    var inlinedCompiledSourcemaps = compiledSourcemaps.js +
        os.EOL +
        convertSourceMap.fromJSON(compiledSourcemaps.v3SourceMap).toComment();
    var combinedSourcemaps = ngAnnotate(inlinedCompiledSourcemaps, {remove: true, add: true, sourcemap: { inline: true, inFile: "sourcemaps.js", sourceRoot: "/source/root/dir" }});
    var combinedSourcemapsSrc = convertSourceMap.removeMapFileComments(combinedSourcemaps.src);
    var combinedSourcemapsMap = convertSourceMap.fromSource(combinedSourcemaps.src).toJSON();
    testSourcemap(originalSourcemaps, combinedSourcemapsSrc, combinedSourcemapsMap, "sourcemaps.annotated.js.map");

    var ngminOriginal = slurp("tests/ngmin-tests/ngmin_original.js");

    console.log("testing adding annotations (imported tests)");
    var ngminAnnotated = ngAnnotate(ngminOriginal, {add: true, regexp: "^myMod"}).src;
    test(slurp("tests/ngmin-tests/ngmin_with_annotations.js"), ngminAnnotated, "ngmin_with_annotations.js");

    console.log("testing removing annotations (imported tests)");
    test(ngminOriginal, ngAnnotate(ngminAnnotated, {remove: true, regexp: "^myMod"}).src, "ngmin_original.js");

    if (fs.existsSync("package.json")) {
        console.log("testing package.json")
        try {
            var json = JSON.parse(slurp("package.json"));
            var substr = JSON.stringify({
                dependencies: json.dependencies,
                devDependencies: json.devDependencies,
            }, null, 4);
            if (/\^/g.test(substr)) {
                console.error("package.json error: shouldn't use the ^ operator");
                console.error(substr);
                process.exit(-1);
            }
        } catch (e) {
            console.error("package.json error: invalid json");
            process.exit(-1);
        }
    }

    if (fs.existsSync("tests/angular.js")) {
        console.log("testing performance");
        var ng1 = String(fs.readFileSync("tests/angular.js"));
        var ng5 = ng1 + ng1 + ng1 + ng1 + ng1;

        time("  ng1 processed in {0} ms", function() { ngAnnotate(ng1, {add: true}) });
        time("  ng1 processed with sourcemaps in {0} ms", function() { ngAnnotate(ng1, {add: true, sourcemap: true}) });
        //time("  ng5 processed in {0} ms", function() { ngAnnotate(ng5, {add: true}) });
        //time("  ng5 processed with sourcemaps in {0} ms", function() { ngAnnotate(ng5, {add: true, sourcemap: true}) });
    }

    console.log("all ok");
}

run(ngAnnotate);

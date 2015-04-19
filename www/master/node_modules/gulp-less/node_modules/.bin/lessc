#!/usr/bin/env node

var path = require('path'),
    fs = require('../lib/less/fs'),
    os = require('os'),
    mkdirp;

var less = require('../lib/less');
var args = process.argv.slice(1);
var options = {
    depends: false,
    compress: false,
    cleancss: false,
    max_line_len: -1,
    optimization: 1,
    silent: false,
    verbose: false,
    lint: false,
    paths: [],
    color: true,
    strictImports: false,
    insecure: false,
    rootpath: '',
    relativeUrls: false,
    ieCompat: true,
    strictMath: false,
    strictUnits: false,
    globalVariables: '',
    modifyVariables: '',
    urlArgs: ''
};
var cleancssOptions = {};
var continueProcessing = true,
    currentErrorcode;

// calling process.exit does not flush stdout always
// so use this to set the exit code
process.on('exit', function() { process.reallyExit(currentErrorcode) });

var checkArgFunc = function(arg, option) {
    if (!option) {
        console.log(arg + " option requires a parameter");
        continueProcessing = false;
        return false;
    }
    return true;
};

var checkBooleanArg = function(arg) {
    var onOff = /^((on|t|true|y|yes)|(off|f|false|n|no))$/i.exec(arg);
    if (!onOff) {
        console.log(" unable to parse "+arg+" as a boolean. use one of on/t/true/y/yes/off/f/false/n/no");
        continueProcessing = false;
        return false;
    }
    return Boolean(onOff[2]);
};

var parseVariableOption = function(option) {
    var parts = option.split('=', 2);
    return '@' + parts[0] + ': ' + parts[1] + ';\n';
};

var warningMessages = "";
var sourceMapFileInline = false;

args = args.filter(function (arg) {
    var match;

    if (match = arg.match(/^-I(.+)$/)) {
        options.paths.push(match[1]);
        return false;
    }

    if (match = arg.match(/^--?([a-z][0-9a-z-]*)(?:=(.*))?$/i)) { arg = match[1] }
    else { return arg }

    switch (arg) {
        case 'v':
        case 'version':
            console.log("lessc " + less.version.join('.') + " (Less Compiler) [JavaScript]");
            continueProcessing = false;
        case 'verbose':
            options.verbose = true;
            break;
        case 's':
        case 'silent':
            options.silent = true;
            break;
        case 'l':
        case 'lint':
            options.lint = true;
            break;
        case 'strict-imports':
            options.strictImports = true;
            break;
        case 'h':
        case 'help':
            require('../lib/less/lessc_helper').printUsage();
            continueProcessing = false;
        case 'x':
        case 'compress':
            options.compress = true;
            break;
        case 'insecure':
            options.insecure = true;
            break;
        case 'M':
        case 'depends':
            options.depends = true;
            break;
        case 'yui-compress':
            warningMessages += "yui-compress option has been removed. ignoring.";
            break;
        case 'clean-css':
            options.cleancss = true;
            break;
        case 'max-line-len':
            if (checkArgFunc(arg, match[2])) {
                options.maxLineLen = parseInt(match[2], 10);
                if (options.maxLineLen <= 0) {
                  options.maxLineLen = -1;
                }
            }
            break;
        case 'no-color':
            options.color = false;
            break;
        case 'no-ie-compat':
            options.ieCompat = false;
            break;
        case 'no-js':
            options.javascriptEnabled = false;
            break;
        case 'include-path':
            if (checkArgFunc(arg, match[2])) {
                options.paths = match[2].split(os.type().match(/Windows/) ? ';' : ':')
                    .map(function(p) {
                        if (p) {
                            return path.resolve(process.cwd(), p);
                        }
                    });
            }
            break;
        case 'O0': options.optimization = 0; break;
        case 'O1': options.optimization = 1; break;
        case 'O2': options.optimization = 2; break;
        case 'line-numbers':
            if (checkArgFunc(arg, match[2])) {
                options.dumpLineNumbers = match[2];
            }
            break;
        case 'source-map':
            if (!match[2]) {
                options.sourceMap = true;
            } else {
                options.sourceMap = match[2];
            }
            break;
        case 'source-map-rootpath':
            if (checkArgFunc(arg, match[2])) {
                options.sourceMapRootpath = match[2];
            }
            break;
        case 'source-map-basepath':
            if (checkArgFunc(arg, match[2])) {
                options.sourceMapBasepath = match[2];
            }
            break;
        case 'source-map-map-inline':
            sourceMapFileInline = true;
            options.sourceMap = true;
            break;
        case 'source-map-less-inline':
            options.outputSourceFiles = true;
            break;
        case 'source-map-url':
            if (checkArgFunc(arg, match[2])) {
                options.sourceMapURL = match[2];
            }
            break;
        case 'rp':
        case 'rootpath':
            if (checkArgFunc(arg, match[2])) {
                options.rootpath = match[2].replace(/\\/g, '/');
            }
            break;
        case "ru":
        case "relative-urls":
            options.relativeUrls = true;
            break;
        case "sm":
        case "strict-math":
            if (checkArgFunc(arg, match[2])) {
                options.strictMath = checkBooleanArg(match[2]);
            }
            break;
        case "su":
        case "strict-units":
            if (checkArgFunc(arg, match[2])) {
                options.strictUnits = checkBooleanArg(match[2]);
            }
            break;
        case "global-var":
            if (checkArgFunc(arg, match[2])) {
                options.globalVariables += parseVariableOption(match[2]);
            }
            break;
        case "modify-var":
            if (checkArgFunc(arg, match[2])) {
                options.modifyVariables += parseVariableOption(match[2]);
            }
            break;
        case "clean-option":
            var cleanOptionArgs = match[2].split(":");
            switch(cleanOptionArgs[0]) {
                case "--keep-line-breaks":
                case "-b":
                    cleancssOptions.keepBreaks = true;
                    break;
                case "--s0":
                    cleancssOptions.keepSpecialComments = 0;
                    break;
                case "--s1":
                    cleancssOptions.keepSpecialComments = 1;
                    break;
                case "--skip-advanced":
                    cleancssOptions.noAdvanced = true;
                    break;
                case "--advanced":
                    cleancssOptions.noAdvanced = false;
                    break;
                case "--compatibility":
                    cleancssOptions.compatibility = cleanOptionArgs[1];
                    break;
                case "--rounding-precision":
                    cleancssOptions.roundingPrecision = Number(cleanOptionArgs[1]);
                    break;
                default:
                    console.log("unrecognised clean-css option '" + cleanOptionArgs[0] + "'");
                    console.log("we support only arguments that make sense for less, '--keep-line-breaks', '-b'");
                    console.log("'--s0', '--s1', '--advanced', '--skip-advanced', '--compatibility', '--rounding-precision'");
                    continueProcessing = false;
                    currentErrorcode = 1;
                    break;
            }
            break;
        case 'url-args':
            if (checkArgFunc(arg, match[2])) {
                options.urlArgs = match[2];
            }
            break;
        default:
            require('../lib/less/lessc_helper').printUsage();
            continueProcessing = false;
            currentErrorcode = 1;
            break;
    }
});

if (!continueProcessing) {
    return;
}

var input = args[1];
var inputbase = args[1];
if (input && input != '-') {
    input = path.resolve(process.cwd(), input);
}
var output = args[2];
var outputbase = args[2];
if (output) {
    options.sourceMapOutputFilename = output;
    output = path.resolve(process.cwd(), output);
    if (warningMessages) {
        console.log(warningMessages);
    }
}

options.sourceMapBasepath = options.sourceMapBasepath || (input ? path.dirname(input) : process.cwd());

if (options.sourceMap === true) {
    if (!output && !sourceMapFileInline) {
        console.log("the sourcemap option only has an optional filename if the css filename is given");
        return;
    }
    options.sourceMapFullFilename = options.sourceMapOutputFilename + ".map";
    options.sourceMap = path.basename(options.sourceMapFullFilename);
}

if (options.cleancss && options.sourceMap) {
    console.log("the cleancss option is not compatible with sourcemap support at the moment. See Issue #1656");
    return;
}

if (! input) {
    console.log("lessc: no input files");
    console.log("");
    require('../lib/less/lessc_helper').printUsage();
    currentErrorcode = 1;
    return;
}

var ensureDirectory = function (filepath) {
    var dir = path.dirname(filepath),
        cmd,
        existsSync = fs.existsSync || path.existsSync;
    if (!existsSync(dir)) {
        if (mkdirp === undefined) {
            try {mkdirp = require('mkdirp');}
            catch(e) { mkdirp = null; }
        }
        cmd = mkdirp && mkdirp.sync || fs.mkdirSync;
        cmd(dir);
    }
};

if (options.depends) {
    if (!outputbase) {
        console.log("option --depends requires an output path to be specified");
        return;
    }
    process.stdout.write(outputbase + ": ");
}

if (!sourceMapFileInline) {
    var writeSourceMap = function(output) {
        var filename = options.sourceMapFullFilename || options.sourceMap;
        ensureDirectory(filename);
        fs.writeFileSync(filename, output, 'utf8');
    };
}

var parseLessFile = function (e, data) {
    if (e) {
        console.log("lessc: " + e.message);
        currentErrorcode = 1;
        return;
    }

    data = options.globalVariables + data + '\n' + options.modifyVariables;

    options.paths = [path.dirname(input)].concat(options.paths);
    options.filename = input;

    var parser = new(less.Parser)(options);
    parser.parse(data, function (err, tree) {
        if (err) {
            less.writeError(err, options);
            currentErrorcode = 1;
            return;
        } else if (options.depends) {
            for(var file in parser.imports.files) {
                process.stdout.write(file + " ")
            }
            console.log("");
        } else {
            try {
	        if (options.lint) { writeSourceMap = function() {} }
                var css = tree.toCSS({
                    silent: options.silent,
                    verbose: options.verbose,
                    ieCompat: options.ieCompat,
                    compress: options.compress,
                    cleancss: options.cleancss,
                    cleancssOptions: cleancssOptions,
                    sourceMap: Boolean(options.sourceMap),
                    sourceMapFilename: options.sourceMap,
                    sourceMapURL: options.sourceMapURL,
                    sourceMapOutputFilename: options.sourceMapOutputFilename,
                    sourceMapBasepath: options.sourceMapBasepath,
                    sourceMapRootpath: options.sourceMapRootpath || "",
                    outputSourceFiles: options.outputSourceFiles,
                    writeSourceMap: writeSourceMap,
                    maxLineLen: options.maxLineLen,
                    strictMath: options.strictMath,
                    strictUnits: options.strictUnits,
                    urlArgs: options.urlArgs
                });
		if(!options.lint) {
                    if (output) {
                        ensureDirectory(output);
                        fs.writeFileSync(output, css, 'utf8');
                        if (options.verbose) {
                            console.log('lessc: wrote ' + output);
                        }
                    } else {
                        process.stdout.write(css);
		    }
		}
            } catch (e) {
                less.writeError(e, options);
                currentErrorcode = 2;
                return;
            }
        }
    });
};

if (input != '-') {
    fs.readFile(input, 'utf8', parseLessFile);
} else {
    process.stdin.resume();
    process.stdin.setEncoding('utf8');

    var buffer = '';
    process.stdin.on('data', function(data) {
        buffer += data;
    });

    process.stdin.on('end', function() {
        parseLessFile(false, buffer);
    });
}

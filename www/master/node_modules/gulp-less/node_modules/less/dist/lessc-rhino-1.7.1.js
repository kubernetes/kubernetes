/* Less.js v1.7.1 RHINO | Copyright (c) 2009-2014, Alexis Sellier <self@cloudhead.net> */

/*global name:true, less, loadStyleSheet, os */

function formatError(ctx, options) {
    options = options || {};

    var message = "";
    var extract = ctx.extract;
    var error = [];

//    var stylize = options.color ? require('./lessc_helper').stylize : function (str) { return str; };
    var stylize = function (str) { return str; };

    // only output a stack if it isn't a less error
    if (ctx.stack && !ctx.type) { return stylize(ctx.stack, 'red'); }

    if (!ctx.hasOwnProperty('index') || !extract) {
        return ctx.stack || ctx.message;
    }

    if (typeof(extract[0]) === 'string') {
        error.push(stylize((ctx.line - 1) + ' ' + extract[0], 'grey'));
    }

    if (typeof(extract[1]) === 'string') {
        var errorTxt = ctx.line + ' ';
        if (extract[1]) {
            errorTxt += extract[1].slice(0, ctx.column) +
                    stylize(stylize(stylize(extract[1][ctx.column], 'bold') +
                            extract[1].slice(ctx.column + 1), 'red'), 'inverse');
        }
        error.push(errorTxt);
    }

    if (typeof(extract[2]) === 'string') {
        error.push(stylize((ctx.line + 1) + ' ' + extract[2], 'grey'));
    }
    error = error.join('\n') + stylize('', 'reset') + '\n';

    message += stylize(ctx.type + 'Error: ' + ctx.message, 'red');
    if (ctx.filename) {
     message += stylize(' in ', 'red') + ctx.filename +
            stylize(' on line ' + ctx.line + ', column ' + (ctx.column + 1) + ':', 'grey');
    }

    message += '\n' + error;

    if (ctx.callLine) {
        message += stylize('from ', 'red') + (ctx.filename || '') + '/n';
        message += stylize(ctx.callLine, 'grey') + ' ' + ctx.callExtract + '/n';
    }

    return message;
}

function writeError(ctx, options) {
    options = options || {};
    if (options.silent) { return; }
    var message = formatError(ctx, options);
    throw new Error(message);
}

function loadStyleSheet(sheet, callback, reload, remaining) {
    var endOfPath = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\')),
        sheetName = name.slice(0, endOfPath + 1) + sheet.href,
        contents = sheet.contents || {},
        input = readFile(sheetName);

    input = input.replace(/^\xEF\xBB\xBF/, '');
        
    contents[sheetName] = input;
        
    var parser = new less.Parser({
        paths: [sheet.href.replace(/[\w\.-]+$/, '')],
        contents: contents
    });
    parser.parse(input, function (e, root) {
        if (e) {
            return writeError(e);
        }
        try {
            callback(e, root, input, sheet, { local: false, lastModified: 0, remaining: remaining }, sheetName);
        } catch(e) {
            writeError(e);
        }
    });
}

less.Parser.fileLoader = function (file, currentFileInfo, callback, env) {

    var href = file;
    if (currentFileInfo && currentFileInfo.currentDirectory && !/^\//.test(file)) {
        href = less.modules.path.join(currentFileInfo.currentDirectory, file);
    }

    var path = less.modules.path.dirname(href);

    var newFileInfo = {
        currentDirectory: path + '/',
        filename: href
    };

    if (currentFileInfo) {
        newFileInfo.entryPath = currentFileInfo.entryPath;
        newFileInfo.rootpath = currentFileInfo.rootpath;
        newFileInfo.rootFilename = currentFileInfo.rootFilename;
        newFileInfo.relativeUrls = currentFileInfo.relativeUrls;
    } else {
        newFileInfo.entryPath = path;
        newFileInfo.rootpath = less.rootpath || path;
        newFileInfo.rootFilename = href;
        newFileInfo.relativeUrls = env.relativeUrls;
    }

    var j = file.lastIndexOf('/');
    if(newFileInfo.relativeUrls && !/^(?:[a-z-]+:|\/)/.test(file) && j != -1) {
        var relativeSubDirectory = file.slice(0, j+1);
        newFileInfo.rootpath = newFileInfo.rootpath + relativeSubDirectory; // append (sub|sup) directory path of imported file
    }
    newFileInfo.currentDirectory = path;
    newFileInfo.filename = href;

    var data = null;
    try {
        data = readFile(href);
    } catch (e) {
        callback({ type: 'File', message: "'" + less.modules.path.basename(href) + "' wasn't found" });
        return;
    }

    try {
        callback(null, data, href, newFileInfo, { lastModified: 0 });
    } catch (e) {
        callback(e, null, href);
    }
};


function writeFile(filename, content) {
    var fstream = new java.io.FileWriter(filename);
    var out = new java.io.BufferedWriter(fstream);
    out.write(content);
    out.close();
}

// Command line integration via Rhino
(function (args) {

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
        rootpath: '',
        relativeUrls: false,
        ieCompat: true,
        strictMath: false,
        strictUnits: false
    };
    var continueProcessing = true,
            currentErrorcode;

    var checkArgFunc = function(arg, option) {
        if (!option) {
            print(arg + " option requires a parameter");
            continueProcessing = false;
            return false;
        }
        return true;
    };

    var checkBooleanArg = function(arg) {
        var onOff = /^((on|t|true|y|yes)|(off|f|false|n|no))$/i.exec(arg);
        if (!onOff) {
            print(" unable to parse "+arg+" as a boolean. use one of on/t/true/y/yes/off/f/false/n/no");
            continueProcessing = false;
            return false;
        }
        return Boolean(onOff[2]);
    };

    var warningMessages = "";
    var sourceMapFileInline = false;

    args = args.filter(function (arg) {
        var match = arg.match(/^-I(.+)$/);
        
        if (match) {
            options.paths.push(match[1]);
            return false;
        }
        
        match = arg.match(/^--?([a-z][0-9a-z-]*)(?:=(.*))?$/i);
        if (match) { arg = match[1]; } // was (?:=([^\s]*)), check!
        else { return arg; }

        switch (arg) {
            case 'v':
            case 'version':
                console.log("lessc " + less.version.join('.') + " (Less Compiler) [JavaScript]");
                continueProcessing = false;
                break;
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
                    //TODO
//                require('../lib/less/lessc_helper').printUsage();
                continueProcessing = false;
                break;
            case 'x':
            case 'compress':
                options.compress = true;
                break;
            case 'M':
            case 'depends':
                options.depends = true;
                break;
            case 'yui-compress':
                warningMessages += "yui-compress option has been removed. assuming clean-css.";
                options.cleancss = true;
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
//                                    return path.resolve(process.cwd(), p);
                                    return p;
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
            case 'source-map-output-map-file':
                if (checkArgFunc(arg, match[2])) {
                    options.writeSourceMap = function(sourceMapContent) {
                         writeFile(match[2], sourceMapContent);
                    };
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
            default:
                console.log('invalid option ' + arg);
                continueProcessing = false;
        }
    });

    if (!continueProcessing) {
        return;
    }

    var name = args[0];
    if (name && name != '-') {
//        name = path.resolve(process.cwd(), name);
    }
    var output = args[1];
    var outputbase = args[1];
    if (output) {
        options.sourceMapOutputFilename = output;
//        output = path.resolve(process.cwd(), output);
        if (warningMessages) {
            console.log(warningMessages);
        }
    }

//  options.sourceMapBasepath = process.cwd();
//    options.sourceMapBasepath = '';

    if (options.sourceMap === true) {
        console.log("output: " + output);
        if (!output && !sourceMapFileInline) {
            console.log("the sourcemap option only has an optional filename if the css filename is given");
            return;
        }
        options.sourceMapFullFilename = options.sourceMapOutputFilename + ".map";
        options.sourceMap = less.modules.path.basename(options.sourceMapFullFilename);
    } else if (options.sourceMap) {
        options.sourceMapOutputFilename = options.sourceMap;
    }
    

    if (!name) {
        console.log("lessc: no inout files");
        console.log("");
        // TODO
//        require('../lib/less/lessc_helper').printUsage();
        currentErrorcode = 1;
        return;
    }

//    var ensureDirectory = function (filepath) {
//        var dir = path.dirname(filepath),
//                cmd,
//                existsSync = fs.existsSync || path.existsSync;
//        if (!existsSync(dir)) {
//            if (mkdirp === undefined) {
//                try {mkdirp = require('mkdirp');}
//                catch(e) { mkdirp = null; }
//            }
//            cmd = mkdirp && mkdirp.sync || fs.mkdirSync;
//            cmd(dir);
//        }
//    };

    if (options.depends) {
        if (!outputbase) {
            console.log("option --depends requires an output path to be specified");
            return;
        }
        console.log(outputbase + ": ");
    }

    if (!name) {
        console.log('No files present in the fileset');
        quit(1);
    }

    var input = null;
    try {
        input = readFile(name, 'utf-8');

    } catch (e) {
        console.log('lesscss: couldn\'t open file ' + name);
        quit(1);
    }

    options.filename = name;
    var result;
    try {
        var parser = new less.Parser(options);
        parser.parse(input, function (e, root) {
            if (e) {
                writeError(e, options);
                quit(1);
            } else {
                result = root.toCSS(options);
                if (output) {
                    writeFile(output, result);
                    console.log("Written to " + output);
                } else {
                    print(result);
                }
                quit(0);
            }
        });
    }
    catch(e) {
        writeError(e, options);
        quit(1);
    }
}(arguments));

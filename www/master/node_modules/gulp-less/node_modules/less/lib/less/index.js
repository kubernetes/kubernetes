var path = require('path'),
    url = require('url'),
    request,
    fs = require('./fs');

var less = {
    version: [1, 7, 5],
    Parser: require('./parser').Parser,
    tree: require('./tree'),
    render: function (input, options, callback) {
        options = options || {};

        if (typeof(options) === 'function') {
            callback = options;
            options = {};
        }

        var parser = new(less.Parser)(options),
            ee;

        if (callback) {
            parser.parse(input, function (e, root) {
                if (e) { callback(e); return; }
                var css;
                try {
                    css = root && root.toCSS && root.toCSS(options);
                }
                catch (err) { callback(err); return; }
                callback(null, css);
            }, options);
        } else {
            ee = new (require('events').EventEmitter)();

            process.nextTick(function () {
                parser.parse(input, function (e, root) {
                    if (e) { return ee.emit('error', e); }
                    try { ee.emit('success', root.toCSS(options)); }
                    catch (err) { ee.emit('error', err); }
                }, options);
            });
            return ee;
        }
    },
    formatError: function(ctx, options) {
        options = options || {};

        var message = "";
        var extract = ctx.extract;
        var error = [];
        var stylize = options.color ? require('./lessc_helper').stylize : function (str) { return str; };

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
    },
    writeError: function (ctx, options) {
        options = options || {};
        if (options.silent) { return; }
        console.error(less.formatError(ctx, options));
    }
};

require('./tree/color');
require('./tree/directive');
require('./tree/detached-ruleset');
require('./tree/operation');
require('./tree/dimension');
require('./tree/keyword');
require('./tree/variable');
require('./tree/ruleset');
require('./tree/element');
require('./tree/selector');
require('./tree/quoted');
require('./tree/expression');
require('./tree/rule');
require('./tree/call');
require('./tree/url');
require('./tree/alpha');
require('./tree/import');
require('./tree/mixin');
require('./tree/comment');
require('./tree/anonymous');
require('./tree/value');
require('./tree/javascript');
require('./tree/assignment');
require('./tree/condition');
require('./tree/paren');
require('./tree/media');
require('./tree/unicode-descriptor');
require('./tree/negative');
require('./tree/extend');
require('./tree/ruleset-call');


var isUrlRe = /^(?:https?:)?\/\//i;

less.Parser.fileLoader = function (file, currentFileInfo, callback, env) {
    var pathname, dirname, data,
        newFileInfo = {
            relativeUrls: env.relativeUrls,
            entryPath: currentFileInfo.entryPath,
            rootpath: currentFileInfo.rootpath,
            rootFilename: currentFileInfo.rootFilename
        };

    function handleDataAndCallCallback(data) {
        var j = file.lastIndexOf('/');

        // Pass on an updated rootpath if path of imported file is relative and file
        // is in a (sub|sup) directory
        //
        // Examples:
        // - If path of imported file is 'module/nav/nav.less' and rootpath is 'less/',
        //   then rootpath should become 'less/module/nav/'
        // - If path of imported file is '../mixins.less' and rootpath is 'less/',
        //   then rootpath should become 'less/../'
        if(newFileInfo.relativeUrls && !/^(?:[a-z-]+:|\/)/.test(file) && j != -1) {
            var relativeSubDirectory = file.slice(0, j+1);
            newFileInfo.rootpath = newFileInfo.rootpath + relativeSubDirectory; // append (sub|sup) directory path of imported file
        }
        newFileInfo.currentDirectory = pathname.replace(/[^\\\/]*$/, "");
        newFileInfo.filename = pathname;

        callback(null, data, pathname, newFileInfo);
    }

    var isUrl = isUrlRe.test( file );
    if (isUrl || isUrlRe.test(currentFileInfo.currentDirectory)) {
        if (request === undefined) {
            try { request = require('request'); }
            catch(e) { request = null; }
        }
        if (!request) {
            callback({ type: 'File', message: "optional dependency 'request' required to import over http(s)\n" });
            return;
        }

        var urlStr = isUrl ? file : url.resolve(currentFileInfo.currentDirectory, file),
            urlObj = url.parse(urlStr);

        if (!urlObj.protocol) {
            urlObj.protocol = "http";
            urlStr = urlObj.format();
        }

        request.get({uri: urlStr, strictSSL: !env.insecure }, function (error, res, body) {
            if (error) {
                callback({ type: 'File', message: "resource '" + urlStr + "' gave this Error:\n  "+ error +"\n" });
                return;
            }
            if (res.statusCode === 404) {
                callback({ type: 'File', message: "resource '" + urlStr + "' was not found\n" });
                return;
            }
            if (!body) {
                console.error( 'Warning: Empty body (HTTP '+ res.statusCode + ') returned by "' + urlStr +'"' );
            }
            pathname = urlStr;
            dirname = urlObj.protocol +'//'+ urlObj.host + urlObj.pathname.replace(/[^\/]*$/, '');
            handleDataAndCallCallback(body);
        });
    } else {

        var paths = [currentFileInfo.currentDirectory];
        if (env.paths) paths.push.apply(paths, env.paths);
        if (paths.indexOf('.') === -1) paths.push('.');

        if (env.syncImport) {
            for (var i = 0; i < paths.length; i++) {
                try {
                    pathname = path.join(paths[i], file);
                    fs.statSync(pathname);
                    break;
                } catch (e) {
                    pathname = null;
                }
            }

            if (!pathname) {
                callback({ type: 'File', message: "'" + file + "' wasn't found" });
                return;
            }

            try {
                data = fs.readFileSync(pathname, 'utf-8');
                handleDataAndCallCallback(data);
            } catch (e) {
                callback(e);
            }
        } else {
            (function tryPathIndex(i) {
                if (i < paths.length) {
                    pathname = path.join(paths[i], file);
                    fs.stat(pathname, function (err) {
                        if (err) {
                            tryPathIndex(i + 1);
                        } else {
                            fs.readFile(pathname, 'utf-8', function(e, data) {
                                if (e) { callback(e); return; }

                                // do processing in the next tick to allow
                                // file handling to dispose
                                process.nextTick(function() {
                                    handleDataAndCallCallback(data);
                                });
                            });
                        }
                    });
                } else {
                    callback({ type: 'File', message: "'" + file + "' wasn't found" });
                }
            }(0));
        }
    }
};

require('./env');
require('./functions');
require('./colors');
require('./visitor.js');
require('./import-visitor.js');
require('./extend-visitor.js');
require('./join-selector-visitor.js');
require('./to-css-visitor.js');
require('./source-map-output.js');

for (var k in less) { if (less.hasOwnProperty(k)) { exports[k] = less[k]; }}

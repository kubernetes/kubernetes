/*!
 * Less - Leaner CSS v1.7.4
 * http://lesscss.org
 *
 * Copyright (c) 2009-2014, Alexis Sellier <self@cloudhead.net>
 * Licensed under the Apache v2 License.
 *
 */

 /** * @license Apache v2
 */

(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);throw new Error("Cannot find module '"+o+"'")}var f=n[o]={exports:{}};t[o][0].call(f.exports,function(e){var n=t[o][1][e];return s(n?n:e)},f,f.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
//
// browser.js - client-side engine
//
/*global window, document, location */

var logLevel = {
    debug: 3,
    info: 2,
    errors: 1,
    none: 0
};

var less;

function log(str, level) {
    if (typeof(console) !== 'undefined' && less.logLevel >= level) {
        console.log('less: ' + str);
    }
}

/*
  TODO - options is now hidden - we should expose it on the less object, but not have it "as" the less object
         alternately even have it on environment
         e.g. less.environment.options.fileAsync = true;
         is it weird you do
         less = { fileAsync: true }
         then access as less.environment.options.fileAsync ?
 */

var isFileProtocol = /^(file|chrome(-extension)?|resource|qrc|app):/.test(location.protocol),
    options = window.less,
    environment = require("./environment/browser.js")(options, isFileProtocol, log, logLevel);

window.less = less = require('./non-node-index.js')(environment);

less.env = options.env || (location.hostname == '127.0.0.1' ||
                        location.hostname == '0.0.0.0'   ||
                        location.hostname == 'localhost' ||
                        (location.port &&
                          location.port.length > 0)      ||
                        isFileProtocol                   ? 'development'
                                                         : 'production');

// The amount of logging in the javascript console.
// 3 - Debug, information and errors
// 2 - Information and errors
// 1 - Errors
// 0 - None
// Defaults to 2
less.logLevel = typeof(options.logLevel) != 'undefined' ? options.logLevel : (less.env === 'development' ?  logLevel.debug : logLevel.errors);

// Load styles asynchronously (default: false)
//
// This is set to `false` by default, so that the body
// doesn't start loading before the stylesheets are parsed.
// Setting this to `true` can result in flickering.
//
options.async = options.async || false;
options.fileAsync = options.fileAsync || false;

// Interval between watch polls
less.poll = less.poll || (isFileProtocol ? 1000 : 1500);

//Setup user functions
if (options.functions) {
    less.functions.functionRegistry.addMultiple(options.functions);
}

var dumpLineNumbers = /!dumpLineNumbers:(comments|mediaquery|all)/.exec(location.hash);
if (dumpLineNumbers) {
    less.dumpLineNumbers = dumpLineNumbers[1];
}

var typePattern = /^text\/(x-)?less$/;
var cache = null;

function extractId(href) {
    return href.replace(/^[a-z-]+:\/+?[^\/]+/, '' )  // Remove protocol & domain
        .replace(/^\//,                 '' )  // Remove root /
        .replace(/\.[a-zA-Z]+$/,        '' )  // Remove simple extension
        .replace(/[^\.\w-]+/g,          '-')  // Replace illegal characters
        .replace(/\./g,                 ':'); // Replace dots with colons(for valid id)
}

function errorConsole(e, rootHref) {
    var template = '{line} {content}';
    var filename = e.filename || rootHref;
    var errors = [];
    var content = (e.type || "Syntax") + "Error: " + (e.message || 'There is an error in your .less file') +
        " in " + filename + " ";

    var errorline = function (e, i, classname) {
        if (e.extract[i] !== undefined) {
            errors.push(template.replace(/\{line\}/, (parseInt(e.line, 10) || 0) + (i - 1))
                .replace(/\{class\}/, classname)
                .replace(/\{content\}/, e.extract[i]));
        }
    };

    if (e.extract) {
        errorline(e, 0, '');
        errorline(e, 1, 'line');
        errorline(e, 2, '');
        content += 'on line ' + e.line + ', column ' + (e.column + 1) + ':\n' +
            errors.join('\n');
    } else if (e.stack) {
        content += e.stack;
    }
    log(content, logLevel.errors);
}

function createCSS(styles, sheet, lastModified) {
    // Strip the query-string
    var href = sheet.href || '';

    // If there is no title set, use the filename, minus the extension
    var id = 'less:' + (sheet.title || extractId(href));

    // If this has already been inserted into the DOM, we may need to replace it
    var oldCss = document.getElementById(id);
    var keepOldCss = false;

    // Create a new stylesheet node for insertion or (if necessary) replacement
    var css = document.createElement('style');
    css.setAttribute('type', 'text/css');
    if (sheet.media) {
        css.setAttribute('media', sheet.media);
    }
    css.id = id;

    if (!css.styleSheet) {
        css.appendChild(document.createTextNode(styles));

        // If new contents match contents of oldCss, don't replace oldCss
        keepOldCss = (oldCss !== null && oldCss.childNodes.length > 0 && css.childNodes.length > 0 &&
            oldCss.firstChild.nodeValue === css.firstChild.nodeValue);
    }

    var head = document.getElementsByTagName('head')[0];

    // If there is no oldCss, just append; otherwise, only append if we need
    // to replace oldCss with an updated stylesheet
    if (oldCss === null || keepOldCss === false) {
        var nextEl = sheet && sheet.nextSibling || null;
        if (nextEl) {
            nextEl.parentNode.insertBefore(css, nextEl);
        } else {
            head.appendChild(css);
        }
    }
    if (oldCss && keepOldCss === false) {
        oldCss.parentNode.removeChild(oldCss);
    }

    // For IE.
    // This needs to happen *after* the style element is added to the DOM, otherwise IE 7 and 8 may crash.
    // See http://social.msdn.microsoft.com/Forums/en-US/7e081b65-878a-4c22-8e68-c10d39c2ed32/internet-explorer-crashes-appending-style-element-to-head
    if (css.styleSheet) {
        try {
            css.styleSheet.cssText = styles;
        } catch (e) {
            throw new(Error)("Couldn't reassign styleSheet.cssText.");
        }
    }

    // Don't update the local store if the file wasn't modified
    if (lastModified && cache) {
        log('saving ' + href + ' to cache.', logLevel.info);
        try {
            cache.setItem(href, styles);
            cache.setItem(href + ':timestamp', lastModified);
        } catch(e) {
            //TODO - could do with adding more robust error handling
            log('failed to save', logLevel.errors);
        }
    }
}

function postProcessCSS(styles) {
    if (options.postProcessor && typeof options.postProcessor === 'function') {
        styles = options.postProcessor.call(styles, styles) || styles;
    }
    return styles;
}

function errorHTML(e, rootHref) {
    var id = 'less-error-message:' + extractId(rootHref || "");
    var template = '<li><label>{line}</label><pre class="{class}">{content}</pre></li>';
    var elem = document.createElement('div'), timer, content, errors = [];
    var filename = e.filename || rootHref;
    var filenameNoPath = filename.match(/([^\/]+(\?.*)?)$/)[1];

    elem.id        = id;
    elem.className = "less-error-message";

    content = '<h3>'  + (e.type || "Syntax") + "Error: " + (e.message || 'There is an error in your .less file') +
        '</h3>' + '<p>in <a href="' + filename   + '">' + filenameNoPath + "</a> ";

    var errorline = function (e, i, classname) {
        if (e.extract[i] !== undefined) {
            errors.push(template.replace(/\{line\}/, (parseInt(e.line, 10) || 0) + (i - 1))
                .replace(/\{class\}/, classname)
                .replace(/\{content\}/, e.extract[i]));
        }
    };

    if (e.extract) {
        errorline(e, 0, '');
        errorline(e, 1, 'line');
        errorline(e, 2, '');
        content += 'on line ' + e.line + ', column ' + (e.column + 1) + ':</p>' +
            '<ul>' + errors.join('') + '</ul>';
    } else if (e.stack) {
        content += '<br/>' + e.stack.split('\n').slice(1).join('<br/>');
    }
    elem.innerHTML = content;

    // CSS for error messages
    createCSS([
        '.less-error-message ul, .less-error-message li {',
        'list-style-type: none;',
        'margin-right: 15px;',
        'padding: 4px 0;',
        'margin: 0;',
        '}',
        '.less-error-message label {',
        'font-size: 12px;',
        'margin-right: 15px;',
        'padding: 4px 0;',
        'color: #cc7777;',
        '}',
        '.less-error-message pre {',
        'color: #dd6666;',
        'padding: 4px 0;',
        'margin: 0;',
        'display: inline-block;',
        '}',
        '.less-error-message pre.line {',
        'color: #ff0000;',
        '}',
        '.less-error-message h3 {',
        'font-size: 20px;',
        'font-weight: bold;',
        'padding: 15px 0 5px 0;',
        'margin: 0;',
        '}',
        '.less-error-message a {',
        'color: #10a',
        '}',
        '.less-error-message .error {',
        'color: red;',
        'font-weight: bold;',
        'padding-bottom: 2px;',
        'border-bottom: 1px dashed red;',
        '}'
    ].join('\n'), { title: 'error-message' });

    elem.style.cssText = [
        "font-family: Arial, sans-serif",
        "border: 1px solid #e00",
        "background-color: #eee",
        "border-radius: 5px",
        "-webkit-border-radius: 5px",
        "-moz-border-radius: 5px",
        "color: #e00",
        "padding: 15px",
        "margin-bottom: 15px"
    ].join(';');

    if (less.env == 'development') {
        timer = setInterval(function () {
            if (document.body) {
                if (document.getElementById(id)) {
                    document.body.replaceChild(elem, document.getElementById(id));
                } else {
                    document.body.insertBefore(elem, document.body.firstChild);
                }
                clearInterval(timer);
            }
        }, 10);
    }
}

function error(e, rootHref) {
    if (!options.errorReporting || options.errorReporting === "html") {
        errorHTML(e, rootHref);
    } else if (options.errorReporting === "console") {
        errorConsole(e, rootHref);
    } else if (typeof options.errorReporting === 'function') {
        options.errorReporting("add", e, rootHref);
    }
}

function removeErrorHTML(path) {
    var node = document.getElementById('less-error-message:' + extractId(path));
    if (node) {
        node.parentNode.removeChild(node);
    }
}

function removeErrorConsole(path) {
    //no action
}

function removeError(path) {
    if (!options.errorReporting || options.errorReporting === "html") {
        removeErrorHTML(path);
    } else if (options.errorReporting === "console") {
        removeErrorConsole(path);
    } else if (typeof options.errorReporting === 'function') {
        options.errorReporting("remove", path);
    }
}

function loadStyles(modifyVars) {
    var styles = document.getElementsByTagName('style'),
        style;
    for (var i = 0; i < styles.length; i++) {
        style = styles[i];
        if (style.type.match(typePattern)) {
            var env = new less.contexts.parseEnv(options),
                lessText = style.innerHTML || '';
            env.filename = document.location.href.replace(/#.*$/, '');

            if (modifyVars || options.globalVars) {
                env.useFileCache = true;
            }

            /*jshint loopfunc:true */
            // use closure to store current value of i
            var callback = (function(style) {
                return function (e, cssAST) {
                    if (e) {
                        return error(e, "inline");
                    }
                    var css = cssAST.toCSS(options);
                    style.type = 'text/css';
                    if (style.styleSheet) {
                        style.styleSheet.cssText = css;
                    } else {
                        style.innerHTML = css;
                    }
                };
            })(style);
            new(less.Parser)(env).parse(lessText, callback, {globalVars: options.globalVars, modifyVars: modifyVars});
        }
    }
}

function loadStyleSheet(sheet, callback, reload, remaining, modifyVars) {

    var env = new less.contexts.parseEnv(options);
    env.mime = sheet.type;

    if (modifyVars || options.globalVars) {
        env.useFileCache = true;
    }

    less.environment.loadFile(env, sheet.href, null, function loadInitialFileCallback(e, data, path, webInfo) {

        var newFileInfo = {
            currentDirectory: less.environment.getPath(env, path),
            filename: path,
            rootFilename: path,
            relativeUrls: env.relativeUrls};

        newFileInfo.entryPath = newFileInfo.currentDirectory;
        newFileInfo.rootpath = env.rootpath || newFileInfo.currentDirectory;

        if (webInfo) {
            webInfo.remaining = remaining;

            var css       = cache && cache.getItem(path),
                timestamp = cache && cache.getItem(path + ':timestamp');

            if (!reload && timestamp && webInfo.lastModified &&
                (new(Date)(webInfo.lastModified).valueOf() ===
                    new(Date)(timestamp).valueOf())) {
                // Use local copy
                createCSS(css, sheet);
                webInfo.local = true;
                callback(null, null, data, sheet, webInfo, path);
                return;
            }
        }

        //TODO add tests around how this behaves when reloading
        removeError(path);

        if (data) {
            env.currentFileInfo = newFileInfo;
            new(less.Parser)(env).parse(data, function (e, root) {
                if (e) { return callback(e, null, null, sheet); }
                try {
                    callback(e, root, data, sheet, webInfo, path);
                } catch (e) {
                    callback(e, null, null, sheet);
                }
            }, {modifyVars: modifyVars, globalVars: options.globalVars});
        } else {
            callback(e, null, null, sheet, webInfo, path);
        }
    }, env, modifyVars);
}

function loadStyleSheets(callback, reload, modifyVars) {
    for (var i = 0; i < less.sheets.length; i++) {
        loadStyleSheet(less.sheets[i], callback, reload, less.sheets.length - (i + 1), modifyVars);
    }
}

function initRunningMode(){
    if (less.env === 'development') {
        less.watchTimer = setInterval(function () {
            if (less.watchMode) {
                loadStyleSheets(function (e, root, _, sheet, env) {
                    if (e) {
                        error(e, sheet.href);
                    } else if (root) {
                        var styles = root.toCSS(less);
                        styles = postProcessCSS(styles);
                        createCSS(styles, sheet, env.lastModified);
                    }
                });
            }
        }, options.poll);
    }
}

//
// Watch mode
//
less.watch   = function () {
    if (!less.watchMode ){
        less.env = 'development';
         initRunningMode();
    }
    this.watchMode = true;
    return true;
};

less.unwatch = function () {clearInterval(less.watchTimer); this.watchMode = false; return false; };

if (/!watch/.test(location.hash)) {
    less.watch();
}

if (less.env != 'development') {
    try {
        cache = (typeof(window.localStorage) === 'undefined') ? null : window.localStorage;
    } catch (_) {}
}

//
// Get all <link> tags with the 'rel' attribute set to "stylesheet/less"
//
var links = document.getElementsByTagName('link');

less.sheets = [];

for (var i = 0; i < links.length; i++) {
    if (links[i].rel === 'stylesheet/less' || (links[i].rel.match(/stylesheet/) &&
       (links[i].type.match(typePattern)))) {
        less.sheets.push(links[i]);
    }
}

//
// With this function, it's possible to alter variables and re-render
// CSS without reloading less-files
//
less.modifyVars = function(record) {
    less.refresh(false, record);
};

less.refresh = function (reload, modifyVars) {
    var startTime, endTime;
    startTime = endTime = new Date();

    loadStyleSheets(function (e, root, _, sheet, env) {
        if (e) {
            return error(e, sheet.href);
        }
        if (env.local) {
            log("loading " + sheet.href + " from cache.", logLevel.info);
        } else {
            log("parsed " + sheet.href + " successfully.", logLevel.debug);
            var styles = root.toCSS(options);
            styles = postProcessCSS(styles);
            createCSS(styles, sheet, env.lastModified);
        }
        log("css for " + sheet.href + " generated in " + (new Date() - endTime) + 'ms', logLevel.info);
        if (env.remaining === 0) {
            log("less has finished. css generated in " + (new Date() - startTime) + 'ms', logLevel.info);
        }
        endTime = new Date();
    }, reload, modifyVars);

    loadStyles(modifyVars);
};

less.refreshStyles = loadStyles;

less.refresh(less.env === 'development');

},{"./environment/browser.js":6,"./non-node-index.js":20}],2:[function(require,module,exports){
var contexts = {};
module.exports = contexts;

var copyFromOriginal = function copyFromOriginal(original, destination, propertiesToCopy) {
    if (!original) { return; }

    for(var i = 0; i < propertiesToCopy.length; i++) {
        if (original.hasOwnProperty(propertiesToCopy[i])) {
            destination[propertiesToCopy[i]] = original[propertiesToCopy[i]];
        }
    }
};

var parseCopyProperties = [
    'paths',            // option - unmodified - paths to search for imports on
    'files',            // list of files that have been imported, used for import-once
    'contents',         // map - filename to contents of all the files
    'contentsIgnoredChars', // map - filename to lines at the begining of each file to ignore
    'relativeUrls',     // option - whether to adjust URL's to be relative
    'rootpath',         // option - rootpath to append to URL's
    'strictImports',    // option -
    'insecure',         // option - whether to allow imports from insecure ssl hosts
    'dumpLineNumbers',  // option - whether to dump line numbers
    'compress',         // option - whether to compress
    'processImports',   // option - whether to process imports. if false then imports will not be imported
    'syncImport',       // option - whether to import synchronously
    'chunkInput',       // option - whether to chunk input. more performant but causes parse issues.
    'mime',             // browser only - mime type for sheet import
    'useFileCache',     // browser only - whether to use the per file session cache
    'currentFileInfo'   // information about the current file - for error reporting and importing and making urls relative etc.
];

//currentFileInfo = {
//  'relativeUrls' - option - whether to adjust URL's to be relative
//  'filename' - full resolved filename of current file
//  'rootpath' - path to append to normal URLs for this node
//  'currentDirectory' - path to the current file, absolute
//  'rootFilename' - filename of the base file
//  'entryPath' - absolute path to the entry file
//  'reference' - whether the file should not be output and only output parts that are referenced

contexts.parseEnv = function(options) {
    copyFromOriginal(options, this, parseCopyProperties);

    if (!this.contents) { this.contents = {}; }
    if (!this.contentsIgnoredChars) { this.contentsIgnoredChars = {}; }
    if (!this.files) { this.files = {}; }

    if (typeof this.paths === "string") { this.paths = [this.paths]; }

    if (!this.currentFileInfo) {
        var filename = (options && options.filename) || "input";
        var entryPath = filename.replace(/[^\/\\]*$/, "");
        if (options) {
            options.filename = null;
        }
        this.currentFileInfo = {
            filename: filename,
            relativeUrls: this.relativeUrls,
            rootpath: (options && options.rootpath) || "",
            currentDirectory: entryPath,
            entryPath: entryPath,
            rootFilename: filename
        };
    }
};

var evalCopyProperties = [
    'silent',         // whether to swallow errors and warnings
    'verbose',        // whether to log more activity
    'compress',       // whether to compress
    'yuicompress',    // whether to compress with the outside tool yui compressor
    'ieCompat',       // whether to enforce IE compatibility (IE8 data-uri)
    'strictMath',     // whether math has to be within parenthesis
    'strictUnits',    // whether units need to evaluate correctly
    'cleancss',       // whether to compress with clean-css
    'sourceMap',      // whether to output a source map
    'importMultiple', // whether we are currently importing multiple copies
    'urlArgs',        // whether to add args into url tokens
    'javascriptEnabled'// option - whether JavaScript is enabled. if undefined, defaults to true
    ];

contexts.evalEnv = function(options, frames) {
    copyFromOriginal(options, this, evalCopyProperties);

    this.frames = frames || [];
};

contexts.evalEnv.prototype.inParenthesis = function () {
    if (!this.parensStack) {
        this.parensStack = [];
    }
    this.parensStack.push(true);
};

contexts.evalEnv.prototype.outOfParenthesis = function () {
    this.parensStack.pop();
};

contexts.evalEnv.prototype.isMathOn = function () {
    return this.strictMath ? (this.parensStack && this.parensStack.length) : true;
};

contexts.evalEnv.prototype.isPathRelative = function (path) {
    return !/^(?:[a-z-]+:|\/)/.test(path);
};

contexts.evalEnv.prototype.normalizePath = function( path ) {
    var
      segments = path.split("/").reverse(),
      segment;

    path = [];
    while (segments.length !== 0 ) {
        segment = segments.pop();
        switch( segment ) {
            case ".":
                break;
            case "..":
                if ((path.length === 0) || (path[path.length - 1] === "..")) {
                    path.push( segment );
                } else {
                    path.pop();
                }
                break;
            default:
                path.push( segment );
                break;
        }
    }

    return path.join("/");
};

//todo - do the same for the toCSS env
//tree.toCSSEnv = function (options) {
//};



},{}],3:[function(require,module,exports){
module.exports = {
    'aliceblue':'#f0f8ff',
    'antiquewhite':'#faebd7',
    'aqua':'#00ffff',
    'aquamarine':'#7fffd4',
    'azure':'#f0ffff',
    'beige':'#f5f5dc',
    'bisque':'#ffe4c4',
    'black':'#000000',
    'blanchedalmond':'#ffebcd',
    'blue':'#0000ff',
    'blueviolet':'#8a2be2',
    'brown':'#a52a2a',
    'burlywood':'#deb887',
    'cadetblue':'#5f9ea0',
    'chartreuse':'#7fff00',
    'chocolate':'#d2691e',
    'coral':'#ff7f50',
    'cornflowerblue':'#6495ed',
    'cornsilk':'#fff8dc',
    'crimson':'#dc143c',
    'cyan':'#00ffff',
    'darkblue':'#00008b',
    'darkcyan':'#008b8b',
    'darkgoldenrod':'#b8860b',
    'darkgray':'#a9a9a9',
    'darkgrey':'#a9a9a9',
    'darkgreen':'#006400',
    'darkkhaki':'#bdb76b',
    'darkmagenta':'#8b008b',
    'darkolivegreen':'#556b2f',
    'darkorange':'#ff8c00',
    'darkorchid':'#9932cc',
    'darkred':'#8b0000',
    'darksalmon':'#e9967a',
    'darkseagreen':'#8fbc8f',
    'darkslateblue':'#483d8b',
    'darkslategray':'#2f4f4f',
    'darkslategrey':'#2f4f4f',
    'darkturquoise':'#00ced1',
    'darkviolet':'#9400d3',
    'deeppink':'#ff1493',
    'deepskyblue':'#00bfff',
    'dimgray':'#696969',
    'dimgrey':'#696969',
    'dodgerblue':'#1e90ff',
    'firebrick':'#b22222',
    'floralwhite':'#fffaf0',
    'forestgreen':'#228b22',
    'fuchsia':'#ff00ff',
    'gainsboro':'#dcdcdc',
    'ghostwhite':'#f8f8ff',
    'gold':'#ffd700',
    'goldenrod':'#daa520',
    'gray':'#808080',
    'grey':'#808080',
    'green':'#008000',
    'greenyellow':'#adff2f',
    'honeydew':'#f0fff0',
    'hotpink':'#ff69b4',
    'indianred':'#cd5c5c',
    'indigo':'#4b0082',
    'ivory':'#fffff0',
    'khaki':'#f0e68c',
    'lavender':'#e6e6fa',
    'lavenderblush':'#fff0f5',
    'lawngreen':'#7cfc00',
    'lemonchiffon':'#fffacd',
    'lightblue':'#add8e6',
    'lightcoral':'#f08080',
    'lightcyan':'#e0ffff',
    'lightgoldenrodyellow':'#fafad2',
    'lightgray':'#d3d3d3',
    'lightgrey':'#d3d3d3',
    'lightgreen':'#90ee90',
    'lightpink':'#ffb6c1',
    'lightsalmon':'#ffa07a',
    'lightseagreen':'#20b2aa',
    'lightskyblue':'#87cefa',
    'lightslategray':'#778899',
    'lightslategrey':'#778899',
    'lightsteelblue':'#b0c4de',
    'lightyellow':'#ffffe0',
    'lime':'#00ff00',
    'limegreen':'#32cd32',
    'linen':'#faf0e6',
    'magenta':'#ff00ff',
    'maroon':'#800000',
    'mediumaquamarine':'#66cdaa',
    'mediumblue':'#0000cd',
    'mediumorchid':'#ba55d3',
    'mediumpurple':'#9370d8',
    'mediumseagreen':'#3cb371',
    'mediumslateblue':'#7b68ee',
    'mediumspringgreen':'#00fa9a',
    'mediumturquoise':'#48d1cc',
    'mediumvioletred':'#c71585',
    'midnightblue':'#191970',
    'mintcream':'#f5fffa',
    'mistyrose':'#ffe4e1',
    'moccasin':'#ffe4b5',
    'navajowhite':'#ffdead',
    'navy':'#000080',
    'oldlace':'#fdf5e6',
    'olive':'#808000',
    'olivedrab':'#6b8e23',
    'orange':'#ffa500',
    'orangered':'#ff4500',
    'orchid':'#da70d6',
    'palegoldenrod':'#eee8aa',
    'palegreen':'#98fb98',
    'paleturquoise':'#afeeee',
    'palevioletred':'#d87093',
    'papayawhip':'#ffefd5',
    'peachpuff':'#ffdab9',
    'peru':'#cd853f',
    'pink':'#ffc0cb',
    'plum':'#dda0dd',
    'powderblue':'#b0e0e6',
    'purple':'#800080',
    'red':'#ff0000',
    'rosybrown':'#bc8f8f',
    'royalblue':'#4169e1',
    'saddlebrown':'#8b4513',
    'salmon':'#fa8072',
    'sandybrown':'#f4a460',
    'seagreen':'#2e8b57',
    'seashell':'#fff5ee',
    'sienna':'#a0522d',
    'silver':'#c0c0c0',
    'skyblue':'#87ceeb',
    'slateblue':'#6a5acd',
    'slategray':'#708090',
    'slategrey':'#708090',
    'snow':'#fffafa',
    'springgreen':'#00ff7f',
    'steelblue':'#4682b4',
    'tan':'#d2b48c',
    'teal':'#008080',
    'thistle':'#d8bfd8',
    'tomato':'#ff6347',
    'turquoise':'#40e0d0',
    'violet':'#ee82ee',
    'wheat':'#f5deb3',
    'white':'#ffffff',
    'whitesmoke':'#f5f5f5',
    'yellow':'#ffff00',
    'yellowgreen':'#9acd32'
};
},{}],4:[function(require,module,exports){
module.exports = {
    colors: require("./colors.js"),
    unitConversions: require("./unit-conversions.js")
};

},{"./colors.js":3,"./unit-conversions.js":5}],5:[function(require,module,exports){
module.exports = {
    length: {
        'm': 1,
        'cm': 0.01,
        'mm': 0.001,
        'in': 0.0254,
        'px': 0.0254 / 96,
        'pt': 0.0254 / 72,
        'pc': 0.0254 / 72 * 12
    },
    duration: {
        's': 1,
        'ms': 0.001
    },
    angle: {
        'rad': 1/(2*Math.PI),
        'deg': 1/360,
        'grad': 1/400,
        'turn': 1
    }
};
},{}],6:[function(require,module,exports){
/*global window, XMLHttpRequest */

module.exports = function(options, isFileProtocol, log, logLevel) {

var fileCache = {};

//TODOS - move log somewhere. pathDiff and doing something similar in node. use pathDiff in the other browser file for the initial load
//        isFileProtocol is global

function getXMLHttpRequest() {
    if (window.XMLHttpRequest && (window.location.protocol !== "file:" || !("ActiveXObject" in window))) {
        return new XMLHttpRequest();
    } else {
        try {
            /*global ActiveXObject */
            return new ActiveXObject("Microsoft.XMLHTTP");
        } catch (e) {
            log("browser doesn't support AJAX.", logLevel.errors);
            return null;
        }
    }
}

return {
    // make generic but overriddable
    warn: function warn(env, msg) {
        console.warn(msg);
    },
    // make generic but overriddable
    getPath: function getPath(env, filename) {
        var j = filename.lastIndexOf('/');
        if (j < 0) {
            j = filename.lastIndexOf('\\');
        }
        if (j < 0) {
            return "";
        }
        return filename.slice(0, j + 1);
    },
    // make generic but overriddable
    isPathAbsolute: function isPathAbsolute(env, filename) {
        return /^(?:[a-z-]+:|\/|\\)/i.test(filename);
    },
    alwaysMakePathsAbsolute: function alwaysMakePathsAbsolute() {
        return true;
    },
    getCleanCSS: function () {
    },
    supportsDataURI: function() {
        return false;
    },
    pathDiff: function pathDiff(url, baseUrl) {
        // diff between two paths to create a relative path

        var urlParts = this.extractUrlParts(url),
            baseUrlParts = this.extractUrlParts(baseUrl),
            i, max, urlDirectories, baseUrlDirectories, diff = "";
        if (urlParts.hostPart !== baseUrlParts.hostPart) {
            return "";
        }
        max = Math.max(baseUrlParts.directories.length, urlParts.directories.length);
        for(i = 0; i < max; i++) {
            if (baseUrlParts.directories[i] !== urlParts.directories[i]) { break; }
        }
        baseUrlDirectories = baseUrlParts.directories.slice(i);
        urlDirectories = urlParts.directories.slice(i);
        for(i = 0; i < baseUrlDirectories.length-1; i++) {
            diff += "../";
        }
        for(i = 0; i < urlDirectories.length-1; i++) {
            diff += urlDirectories[i] + "/";
        }
        return diff;
    },
    join: function join(basePath, laterPath) {
        if (!basePath) {
            return laterPath;
        }
        return this.extractUrlParts(laterPath, basePath).path;
    },
    // helper function, not part of API
    extractUrlParts: function extractUrlParts(url, baseUrl) {
        // urlParts[1] = protocol&hostname || /
        // urlParts[2] = / if path relative to host base
        // urlParts[3] = directories
        // urlParts[4] = filename
        // urlParts[5] = parameters

        var urlPartsRegex = /^((?:[a-z-]+:)?\/+?(?:[^\/\?#]*\/)|([\/\\]))?((?:[^\/\\\?#]*[\/\\])*)([^\/\\\?#]*)([#\?].*)?$/i,
            urlParts = url.match(urlPartsRegex),
            returner = {}, directories = [], i, baseUrlParts;

        if (!urlParts) {
            throw new Error("Could not parse sheet href - '"+url+"'");
        }

        // Stylesheets in IE don't always return the full path
        if (!urlParts[1] || urlParts[2]) {
            baseUrlParts = baseUrl.match(urlPartsRegex);
            if (!baseUrlParts) {
                throw new Error("Could not parse page url - '"+baseUrl+"'");
            }
            urlParts[1] = urlParts[1] || baseUrlParts[1] || "";
            if (!urlParts[2]) {
                urlParts[3] = baseUrlParts[3] + urlParts[3];
            }
        }

        if (urlParts[3]) {
            directories = urlParts[3].replace(/\\/g, "/").split("/");

            // extract out . before .. so .. doesn't absorb a non-directory
            for(i = 0; i < directories.length; i++) {
                if (directories[i] === ".") {
                    directories.splice(i, 1);
                    i -= 1;
                }
            }

            for(i = 0; i < directories.length; i++) {
                if (directories[i] === ".." && i > 0) {
                    directories.splice(i-1, 2);
                    i -= 2;
                }
            }
        }

        returner.hostPart = urlParts[1];
        returner.directories = directories;
        returner.path = urlParts[1] + directories.join("/");
        returner.fileUrl = returner.path + (urlParts[4] || "");
        returner.url = returner.fileUrl + (urlParts[5] || "");
        return returner;
    },
    doXHR: function doXHR(url, type, callback, errback) {

        var xhr = getXMLHttpRequest();
        var async = isFileProtocol ? options.fileAsync : options.async;

        if (typeof(xhr.overrideMimeType) === 'function') {
            xhr.overrideMimeType('text/css');
        }
        log("XHR: Getting '" + url + "'", logLevel.debug);
        xhr.open('GET', url, async);
        xhr.setRequestHeader('Accept', type || 'text/x-less, text/css; q=0.9, */*; q=0.5');
        xhr.send(null);

        function handleResponse(xhr, callback, errback) {
            if (xhr.status >= 200 && xhr.status < 300) {
                callback(xhr.responseText,
                    xhr.getResponseHeader("Last-Modified"));
            } else if (typeof(errback) === 'function') {
                errback(xhr.status, url);
            }
        }

        if (isFileProtocol && !options.fileAsync) {
            if (xhr.status === 0 || (xhr.status >= 200 && xhr.status < 300)) {
                callback(xhr.responseText);
            } else {
                errback(xhr.status, url);
            }
        } else if (async) {
            xhr.onreadystatechange = function () {
                if (xhr.readyState == 4) {
                    handleResponse(xhr, callback, errback);
                }
            };
        } else {
            handleResponse(xhr, callback, errback);
        }
    },
    loadFile: function loadFile(env, filename, currentDirectory, callback) {
        if (currentDirectory && !this.isPathAbsolute(env, filename)) {
            filename = currentDirectory + filename;
        }

        // sheet may be set to the stylesheet for the initial load or a collection of properties including
        // some env variables for imports
        var hrefParts = this.extractUrlParts(filename, window.location.href);
        var href      = hrefParts.url;

        if (env.useFileCache && fileCache[href]) {
            try {
                var lessText = fileCache[href];
                callback(null, lessText, href, { lastModified: new Date() });
            } catch (e) {
                callback(e, null, href);
            }
            return;
        }

        this.doXHR(href, env.mime, function doXHRCallback(data, lastModified) {
            // per file cache
            fileCache[href] = data;

            // Use remote copy (re-parse)
            callback(null, data, href, { lastModified: lastModified });
        }, function doXHRError(status, url) {
            callback({ type: 'File', message: "'" + url + "' wasn't found (" + status + ")" }, null, href);
        });

    }
};

};

},{}],7:[function(require,module,exports){
var Color = require("../tree/color.js"),
    functionRegistry = require("./function-registry.js");

// Color Blending
// ref: http://www.w3.org/TR/compositing-1

function colorBlend(mode, color1, color2) {
    var ab = color1.alpha, cb, // backdrop
        as = color2.alpha, cs, // source
        ar, cr, r = [];        // result

    ar = as + ab * (1 - as);
    for (var i = 0; i < 3; i++) {
        cb = color1.rgb[i] / 255;
        cs = color2.rgb[i] / 255;
        cr = mode(cb, cs);
        if (ar) {
            cr = (as * cs + ab * (cb -
                  as * (cb + cs - cr))) / ar;
        }
        r[i] = cr * 255;
    }

    return new(Color)(r, ar);
}

var colorBlendModeFunctions = {
    multiply: function(cb, cs) {
        return cb * cs;
    },
    screen: function(cb, cs) {
        return cb + cs - cb * cs;
    },
    overlay: function(cb, cs) {
        cb *= 2;
        return (cb <= 1)
            ? colorBlendModeFunctions.multiply(cb, cs)
            : colorBlendModeFunctions.screen(cb - 1, cs);
    },
    softlight: function(cb, cs) {
        var d = 1, e = cb;
        if (cs > 0.5) {
            e = 1;
            d = (cb > 0.25) ? Math.sqrt(cb)
                : ((16 * cb - 12) * cb + 4) * cb;
        }
        return cb - (1 - 2 * cs) * e * (d - cb);
    },
    hardlight: function(cb, cs) {
        return colorBlendModeFunctions.overlay(cs, cb);
    },
    difference: function(cb, cs) {
        return Math.abs(cb - cs);
    },
    exclusion: function(cb, cs) {
        return cb + cs - 2 * cb * cs;
    },

    // non-w3c functions:
    average: function(cb, cs) {
        return (cb + cs) / 2;
    },
    negation: function(cb, cs) {
        return 1 - Math.abs(cb + cs - 1);
    }
};

for (var f in colorBlendModeFunctions) {
    if (colorBlendModeFunctions.hasOwnProperty(f)) {
        colorBlend[f] = colorBlend.bind(null, colorBlendModeFunctions[f]);
    }
}

functionRegistry.addMultiple(colorBlend);

},{"../tree/color.js":31,"./function-registry.js":12}],8:[function(require,module,exports){
var Dimension = require("../tree/dimension.js"),
    Color = require("../tree/color.js"),
    Quoted = require("../tree/quoted.js"),
    Anonymous = require("../tree/anonymous.js"),
    functionRegistry = require("./function-registry.js"),
    colorFunctions;

function clamp(val) {
    return Math.min(1, Math.max(0, val));
}
function hsla(color) {
    return colorFunctions.hsla(color.h, color.s, color.l, color.a);
}
function number(n) {
    if (n instanceof Dimension) {
        return parseFloat(n.unit.is('%') ? n.value / 100 : n.value);
    } else if (typeof(n) === 'number') {
        return n;
    } else {
        throw {
            error: "RuntimeError",
            message: "color functions take numbers as parameters"
        };
    }
}
function scaled(n, size) {
    if (n instanceof Dimension && n.unit.is('%')) {
        return parseFloat(n.value * size / 100);
    } else {
        return number(n);
    }
}
colorFunctions = {
    rgb: function (r, g, b) {
        return colorFunctions.rgba(r, g, b, 1.0);
    },
    rgba: function (r, g, b, a) {
        var rgb = [r, g, b].map(function (c) { return scaled(c, 255); });
        a = number(a);
        return new(Color)(rgb, a);
    },
    hsl: function (h, s, l) {
        return colorFunctions.hsla(h, s, l, 1.0);
    },
    hsla: function (h, s, l, a) {
        function hue(h) {
            h = h < 0 ? h + 1 : (h > 1 ? h - 1 : h);
            if      (h * 6 < 1) { return m1 + (m2 - m1) * h * 6; }
            else if (h * 2 < 1) { return m2; }
            else if (h * 3 < 2) { return m1 + (m2 - m1) * (2/3 - h) * 6; }
            else                { return m1; }
        }

        h = (number(h) % 360) / 360;
        s = clamp(number(s)); l = clamp(number(l)); a = clamp(number(a));

        var m2 = l <= 0.5 ? l * (s + 1) : l + s - l * s;
        var m1 = l * 2 - m2;

        return colorFunctions.rgba(hue(h + 1/3) * 255,
            hue(h)       * 255,
            hue(h - 1/3) * 255,
            a);
    },

    hsv: function(h, s, v) {
        return colorFunctions.hsva(h, s, v, 1.0);
    },

    hsva: function(h, s, v, a) {
        h = ((number(h) % 360) / 360) * 360;
        s = number(s); v = number(v); a = number(a);

        var i, f;
        i = Math.floor((h / 60) % 6);
        f = (h / 60) - i;

        var vs = [v,
            v * (1 - s),
            v * (1 - f * s),
            v * (1 - (1 - f) * s)];
        var perm = [[0, 3, 1],
            [2, 0, 1],
            [1, 0, 3],
            [1, 2, 0],
            [3, 1, 0],
            [0, 1, 2]];

        return colorFunctions.rgba(vs[perm[i][0]] * 255,
            vs[perm[i][1]] * 255,
            vs[perm[i][2]] * 255,
            a);
    },

    hue: function (color) {
        return new(Dimension)(color.toHSL().h);
    },
    saturation: function (color) {
        return new(Dimension)(color.toHSL().s * 100, '%');
    },
    lightness: function (color) {
        return new(Dimension)(color.toHSL().l * 100, '%');
    },
    hsvhue: function(color) {
        return new(Dimension)(color.toHSV().h);
    },
    hsvsaturation: function (color) {
        return new(Dimension)(color.toHSV().s * 100, '%');
    },
    hsvvalue: function (color) {
        return new(Dimension)(color.toHSV().v * 100, '%');
    },
    red: function (color) {
        return new(Dimension)(color.rgb[0]);
    },
    green: function (color) {
        return new(Dimension)(color.rgb[1]);
    },
    blue: function (color) {
        return new(Dimension)(color.rgb[2]);
    },
    alpha: function (color) {
        return new(Dimension)(color.toHSL().a);
    },
    luma: function (color) {
        return new(Dimension)(color.luma() * color.alpha * 100, '%');
    },
    luminance: function (color) {
        var luminance =
            (0.2126 * color.rgb[0] / 255)
                + (0.7152 * color.rgb[1] / 255)
                + (0.0722 * color.rgb[2] / 255);

        return new(Dimension)(luminance * color.alpha * 100, '%');
    },
    saturate: function (color, amount) {
        // filter: saturate(3.2);
        // should be kept as is, so check for color
        if (!color.rgb) {
            return null;
        }
        var hsl = color.toHSL();

        hsl.s += amount.value / 100;
        hsl.s = clamp(hsl.s);
        return hsla(hsl);
    },
    desaturate: function (color, amount) {
        var hsl = color.toHSL();

        hsl.s -= amount.value / 100;
        hsl.s = clamp(hsl.s);
        return hsla(hsl);
    },
    lighten: function (color, amount) {
        var hsl = color.toHSL();

        hsl.l += amount.value / 100;
        hsl.l = clamp(hsl.l);
        return hsla(hsl);
    },
    darken: function (color, amount) {
        var hsl = color.toHSL();

        hsl.l -= amount.value / 100;
        hsl.l = clamp(hsl.l);
        return hsla(hsl);
    },
    fadein: function (color, amount) {
        var hsl = color.toHSL();

        hsl.a += amount.value / 100;
        hsl.a = clamp(hsl.a);
        return hsla(hsl);
    },
    fadeout: function (color, amount) {
        var hsl = color.toHSL();

        hsl.a -= amount.value / 100;
        hsl.a = clamp(hsl.a);
        return hsla(hsl);
    },
    fade: function (color, amount) {
        var hsl = color.toHSL();

        hsl.a = amount.value / 100;
        hsl.a = clamp(hsl.a);
        return hsla(hsl);
    },
    spin: function (color, amount) {
        var hsl = color.toHSL();
        var hue = (hsl.h + amount.value) % 360;

        hsl.h = hue < 0 ? 360 + hue : hue;

        return hsla(hsl);
    },
    //
    // Copyright (c) 2006-2009 Hampton Catlin, Nathan Weizenbaum, and Chris Eppstein
    // http://sass-lang.com
    //
    mix: function (color1, color2, weight) {
        if (!weight) {
            weight = new(Dimension)(50);
        }
        var p = weight.value / 100.0;
        var w = p * 2 - 1;
        var a = color1.toHSL().a - color2.toHSL().a;

        var w1 = (((w * a == -1) ? w : (w + a) / (1 + w * a)) + 1) / 2.0;
        var w2 = 1 - w1;

        var rgb = [color1.rgb[0] * w1 + color2.rgb[0] * w2,
            color1.rgb[1] * w1 + color2.rgb[1] * w2,
            color1.rgb[2] * w1 + color2.rgb[2] * w2];

        var alpha = color1.alpha * p + color2.alpha * (1 - p);

        return new(Color)(rgb, alpha);
    },
    greyscale: function (color) {
        return colorFunctions.desaturate(color, new(Dimension)(100));
    },
    contrast: function (color, dark, light, threshold) {
        // filter: contrast(3.2);
        // should be kept as is, so check for color
        if (!color.rgb) {
            return null;
        }
        if (typeof light === 'undefined') {
            light = colorFunctions.rgba(255, 255, 255, 1.0);
        }
        if (typeof dark === 'undefined') {
            dark = colorFunctions.rgba(0, 0, 0, 1.0);
        }
        //Figure out which is actually light and dark!
        if (dark.luma() > light.luma()) {
            var t = light;
            light = dark;
            dark = t;
        }
        if (typeof threshold === 'undefined') {
            threshold = 0.43;
        } else {
            threshold = number(threshold);
        }
        if (color.luma() < threshold) {
            return light;
        } else {
            return dark;
        }
    },
    argb: function (color) {
        return new(Anonymous)(color.toARGB());
    },
    color: function(c) {
        if ((c instanceof Quoted) &&
            (/^#([a-f0-9]{6}|[a-f0-9]{3})$/i.test(c.value))) {
            return new(Color)(c.value.slice(1));
        }
        if ((c instanceof Color) || (c = Color.fromKeyword(c.value))) {
            c.keyword = undefined;
            return c;
        }
        throw {
            type:    "Argument",
            message: "argument must be a color keyword or 3/6 digit hex e.g. #FFF"
        };
    },
    tint: function(color, amount) {
        return colorFunctions.mix(colorFunctions.rgb(255,255,255), color, amount);
    },
    shade: function(color, amount) {
        return colorFunctions.mix(colorFunctions.rgb(0, 0, 0), color, amount);
    }
};
functionRegistry.addMultiple(colorFunctions);

},{"../tree/anonymous.js":27,"../tree/color.js":31,"../tree/dimension.js":37,"../tree/quoted.js":54,"./function-registry.js":12}],9:[function(require,module,exports){
module.exports = function(environment) {
    var Anonymous = require("../tree/anonymous.js"),
        URL = require("../tree/url.js"),
        functionRegistry = require("./function-registry.js");

    functionRegistry.add("data-uri", function(mimetypeNode, filePathNode) {

        if (!environment.supportsDataURI(this.env)) {
            return new URL(filePathNode || mimetypeNode, this.currentFileInfo).eval(this.env);
        }

        var mimetype = mimetypeNode.value;
        var filePath = (filePathNode && filePathNode.value);

        var useBase64 = false;

        if (arguments.length < 2) {
            filePath = mimetype;
        }

        var fragmentStart = filePath.indexOf('#');
        var fragment = '';
        if (fragmentStart!==-1) {
            fragment = filePath.slice(fragmentStart);
            filePath = filePath.slice(0, fragmentStart);
        }

        if (this.env.isPathRelative(filePath)) {
            if (this.currentFileInfo.relativeUrls) {
                filePath = environment.join(this.currentFileInfo.currentDirectory, filePath);
            } else {
                filePath = environment.join(this.currentFileInfo.entryPath, filePath);
            }
        }

        // detect the mimetype if not given
        if (arguments.length < 2) {

            mimetype = environment.mimeLookup(this.env, filePath);

            // use base 64 unless it's an ASCII or UTF-8 format
            var charset = environment.charsetLookup(this.env, mimetype);
            useBase64 = ['US-ASCII', 'UTF-8'].indexOf(charset) < 0;
            if (useBase64) { mimetype += ';base64'; }
        }
        else {
            useBase64 = /;base64$/.test(mimetype);
        }

        var buf = environment.readFileSync(filePath);

        // IE8 cannot handle a data-uri larger than 32KB. If this is exceeded
        // and the --ieCompat flag is enabled, return a normal url() instead.
        var DATA_URI_MAX_KB = 32,
            fileSizeInKB = parseInt((buf.length / 1024), 10);
        if (fileSizeInKB >= DATA_URI_MAX_KB) {

            if (this.env.ieCompat !== false) {
                if (!this.env.silent) {
                    console.warn("Skipped data-uri embedding of %s because its size (%dKB) exceeds IE8-safe %dKB!", filePath, fileSizeInKB, DATA_URI_MAX_KB);
                }

                return new URL(filePathNode || mimetypeNode, this.currentFileInfo).eval(this.env);
            }
        }

        buf = useBase64 ? buf.toString('base64')
            : encodeURIComponent(buf);

        var uri = "\"data:" + mimetype + ',' + buf + fragment + "\"";
        return new(URL)(new(Anonymous)(uri));
    });
};

},{"../tree/anonymous.js":27,"../tree/url.js":61,"./function-registry.js":12}],10:[function(require,module,exports){
var Keyword = require("../tree/keyword.js"),
    functionRegistry = require("./function-registry.js");

var defaultFunc = {
    eval: function () {
        var v = this.value_, e = this.error_;
        if (e) {
            throw e;
        }
        if (v != null) {
            return v ? Keyword.True : Keyword.False;
        }
    },
    value: function (v) {
        this.value_ = v;
    },
    error: function (e) {
        this.error_ = e;
    },
    reset: function () {
        this.value_ = this.error_ = null;
    }
};

functionRegistry.add("default", defaultFunc.eval.bind(defaultFunc));

module.exports = defaultFunc;

},{"../tree/keyword.js":46,"./function-registry.js":12}],11:[function(require,module,exports){
var functionRegistry = require("./function-registry.js");

var functionCaller = function(name, env, currentFileInfo) {
    this.name = name.toLowerCase();
    this.function = functionRegistry.get(this.name);
    this.env = env;
    this.currentFileInfo = currentFileInfo;
};
functionCaller.prototype.isValid = function() {
    return Boolean(this.function);
};
functionCaller.prototype.call = function(args) {
    return this.function.apply(this, args);
};

module.exports = functionCaller;

},{"./function-registry.js":12}],12:[function(require,module,exports){
module.exports = {
    _data: {},
    add: function(name, func) {
        if (this._data.hasOwnProperty(name)) {
            //TODO warn
        }
        this._data[name] = func;
    },
    addMultiple: function(functions) {
        Object.keys(functions).forEach(
            function(name) {
                this.add(name, functions[name]);
            }.bind(this));
    },
    get: function(name) {
        return this._data[name];
    }
};

},{}],13:[function(require,module,exports){
module.exports = function(environment) {
    var functions = {
        functionRegistry: require("./function-registry.js"),
        functionCaller: require("./function-caller.js")
    };

    //register functions
    require("./default.js");
    require("./color.js");
    require("./color-blending.js");
    require("./data-uri.js")(environment);
    require("./math.js");
    require("./number.js");
    require("./string.js");
    require("./svg.js")(environment);
    require("./types.js");

    return functions;
};

},{"./color-blending.js":7,"./color.js":8,"./data-uri.js":9,"./default.js":10,"./function-caller.js":11,"./function-registry.js":12,"./math.js":14,"./number.js":15,"./string.js":16,"./svg.js":17,"./types.js":18}],14:[function(require,module,exports){
var Dimension = require("../tree/dimension.js"),
    functionRegistry = require("./function-registry.js");

var mathFunctions = {
    // name,  unit
    ceil:  null,
    floor: null,
    sqrt:  null,
    abs:   null,
    tan:   "",
    sin:   "",
    cos:   "",
    atan:  "rad",
    asin:  "rad",
    acos:  "rad"
};

function _math(fn, unit, n) {
    if (!(n instanceof Dimension)) {
        throw { type: "Argument", message: "argument must be a number" };
    }
    if (unit == null) {
        unit = n.unit;
    } else {
        n = n.unify();
    }
    return new(Dimension)(fn(parseFloat(n.value)), unit);
}

for (var f in mathFunctions) {
    if (mathFunctions.hasOwnProperty(f)) {
        mathFunctions[f] = _math.bind(null, Math[f], mathFunctions[f]);
    }
}

mathFunctions.round = function (n, f) {
    var fraction = typeof(f) === "undefined" ? 0 : f.value;
    return _math(function(num) { return num.toFixed(fraction); }, null, n);
};

functionRegistry.addMultiple(mathFunctions);

},{"../tree/dimension.js":37,"./function-registry.js":12}],15:[function(require,module,exports){
var Dimension = require("../tree/dimension.js"),
    Anonymous = require("../tree/anonymous.js"),
    functionRegistry = require("./function-registry.js");

var minMax = function (isMin, args) {
    args = Array.prototype.slice.call(args);
    switch(args.length) {
        case 0: throw { type: "Argument", message: "one or more arguments required" };
    }
    var i, j, current, currentUnified, referenceUnified, unit, unitStatic, unitClone,
        order  = [], // elems only contains original argument values.
        values = {}; // key is the unit.toString() for unified Dimension values,
    // value is the index into the order array.
    for (i = 0; i < args.length; i++) {
        current = args[i];
        if (!(current instanceof Dimension)) {
            if(Array.isArray(args[i].value)) {
                Array.prototype.push.apply(args, Array.prototype.slice.call(args[i].value));
            }
            continue;
        }
        currentUnified = current.unit.toString() === "" && unitClone !== undefined ? new(Dimension)(current.value, unitClone).unify() : current.unify();
        unit = currentUnified.unit.toString() === "" && unitStatic !== undefined ? unitStatic : currentUnified.unit.toString();
        unitStatic = unit !== "" && unitStatic === undefined || unit !== "" && order[0].unify().unit.toString() === "" ? unit : unitStatic;
        unitClone = unit !== "" && unitClone === undefined ? current.unit.toString() : unitClone;
        j = values[""] !== undefined && unit !== "" && unit === unitStatic ? values[""] : values[unit];
        if (j === undefined) {
            if(unitStatic !== undefined && unit !== unitStatic) {
                throw{ type: "Argument", message: "incompatible types" };
            }
            values[unit] = order.length;
            order.push(current);
            continue;
        }
        referenceUnified = order[j].unit.toString() === "" && unitClone !== undefined ? new(Dimension)(order[j].value, unitClone).unify() : order[j].unify();
        if ( isMin && currentUnified.value < referenceUnified.value ||
            !isMin && currentUnified.value > referenceUnified.value) {
            order[j] = current;
        }
    }
    if (order.length == 1) {
        return order[0];
    }
    args = order.map(function (a) { return a.toCSS(this.env); }).join(this.env.compress ? "," : ", ");
    return new(Anonymous)((isMin ? "min" : "max") + "(" + args + ")");
};
functionRegistry.addMultiple({
    min: function () {
        return minMax(true, arguments);
    },
    max: function () {
        return minMax(false, arguments);
    },
    convert: function (val, unit) {
        return val.convertTo(unit.value);
    },
    pi: function () {
        return new(Dimension)(Math.PI);
    },
    mod: function(a, b) {
        return new(Dimension)(a.value % b.value, a.unit);
    },
    pow: function(x, y) {
        if (typeof x === "number" && typeof y === "number") {
            x = new(Dimension)(x);
            y = new(Dimension)(y);
        } else if (!(x instanceof Dimension) || !(y instanceof Dimension)) {
            throw { type: "Argument", message: "arguments must be numbers" };
        }

        return new(Dimension)(Math.pow(x.value, y.value), x.unit);
    },
    percentage: function (n) {
        return new(Dimension)(n.value * 100, '%');
    }
});

},{"../tree/anonymous.js":27,"../tree/dimension.js":37,"./function-registry.js":12}],16:[function(require,module,exports){
var Quoted = require("../tree/quoted.js"),
    Anonymous = require("../tree/anonymous.js"),
    JavaScript = require("../tree/javascript.js"),
    functionRegistry = require("./function-registry.js");

functionRegistry.addMultiple({
    e: function (str) {
        return new(Anonymous)(str instanceof JavaScript ? str.evaluated : str.value);
    },
    escape: function (str) {
        return new(Anonymous)(encodeURI(str.value).replace(/=/g, "%3D").replace(/:/g, "%3A").replace(/#/g, "%23").replace(/;/g, "%3B").replace(/\(/g, "%28").replace(/\)/g, "%29"));
    },
    replace: function (string, pattern, replacement, flags) {
        var result = string.value;

        result = result.replace(new RegExp(pattern.value, flags ? flags.value : ''), replacement.value);
        return new(Quoted)(string.quote || '', result, string.escaped);
    },
    '%': function (string /* arg, arg, ...*/) {
        var args = Array.prototype.slice.call(arguments, 1),
            result = string.value;

        for (var i = 0; i < args.length; i++) {
            /*jshint loopfunc:true */
            result = result.replace(/%[sda]/i, function(token) {
                var value = token.match(/s/i) ? args[i].value : args[i].toCSS();
                return token.match(/[A-Z]$/) ? encodeURIComponent(value) : value;
            });
        }
        result = result.replace(/%%/g, '%');
        return new(Quoted)(string.quote || '', result, string.escaped);
    }
});

},{"../tree/anonymous.js":27,"../tree/javascript.js":44,"../tree/quoted.js":54,"./function-registry.js":12}],17:[function(require,module,exports){
module.exports = function(environment) {
    var Dimension = require("../tree/dimension.js"),
        Color = require("../tree/color.js"),
        Anonymous = require("../tree/anonymous.js"),
        URL = require("../tree/url.js"),
        functionRegistry = require("./function-registry.js");

    functionRegistry.add("svg-gradient", function(direction) {

        function throwArgumentDescriptor() {
            throw { type: "Argument", message: "svg-gradient expects direction, start_color [start_position], [color position,]..., end_color [end_position]" };
        }

        if (arguments.length < 3) {
            throwArgumentDescriptor();
        }
        var stops = Array.prototype.slice.call(arguments, 1),
            gradientDirectionSvg,
            gradientType = "linear",
            rectangleDimension = 'x="0" y="0" width="1" height="1"',
            useBase64 = true,
            renderEnv = {compress: false},
            returner,
            directionValue = direction.toCSS(renderEnv),
            i, color, position, positionValue, alpha;

        switch (directionValue) {
            case "to bottom":
                gradientDirectionSvg = 'x1="0%" y1="0%" x2="0%" y2="100%"';
                break;
            case "to right":
                gradientDirectionSvg = 'x1="0%" y1="0%" x2="100%" y2="0%"';
                break;
            case "to bottom right":
                gradientDirectionSvg = 'x1="0%" y1="0%" x2="100%" y2="100%"';
                break;
            case "to top right":
                gradientDirectionSvg = 'x1="0%" y1="100%" x2="100%" y2="0%"';
                break;
            case "ellipse":
            case "ellipse at center":
                gradientType = "radial";
                gradientDirectionSvg = 'cx="50%" cy="50%" r="75%"';
                rectangleDimension = 'x="-50" y="-50" width="101" height="101"';
                break;
            default:
                throw { type: "Argument", message: "svg-gradient direction must be 'to bottom', 'to right', 'to bottom right', 'to top right' or 'ellipse at center'" };
        }
        returner = '<?xml version="1.0" ?>' +
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="100%" height="100%" viewBox="0 0 1 1" preserveAspectRatio="none">' +
            '<' + gradientType + 'Gradient id="gradient" gradientUnits="userSpaceOnUse" ' + gradientDirectionSvg + '>';

        for (i = 0; i < stops.length; i+= 1) {
            if (stops[i].value) {
                color = stops[i].value[0];
                position = stops[i].value[1];
            } else {
                color = stops[i];
                position = undefined;
            }

            if (!(color instanceof Color) || (!((i === 0 || i+1 === stops.length) && position === undefined) && !(position instanceof Dimension))) {
                throwArgumentDescriptor();
            }
            positionValue = position ? position.toCSS(renderEnv) : i === 0 ? "0%" : "100%";
            alpha = color.alpha;
            returner += '<stop offset="' + positionValue + '" stop-color="' + color.toRGB() + '"' + (alpha < 1 ? ' stop-opacity="' + alpha + '"' : '') + '/>';
        }
        returner += '</' + gradientType + 'Gradient>' +
            '<rect ' + rectangleDimension + ' fill="url(#gradient)" /></svg>';

        if (useBase64) {
            try {
                returner = environment.encodeBase64(this.env, returner);
            } catch(e) {
                useBase64 = false;
            }
        }

        returner = "'data:image/svg+xml" + (useBase64 ? ";base64" : "") + "," + returner + "'";
        return new(URL)(new(Anonymous)(returner));
    });
};

},{"../tree/anonymous.js":27,"../tree/color.js":31,"../tree/dimension.js":37,"../tree/url.js":61,"./function-registry.js":12}],18:[function(require,module,exports){
var Keyword = require("../tree/keyword.js"),
    Dimension = require("../tree/dimension.js"),
    Color = require("../tree/color.js"),
    Quoted = require("../tree/quoted.js"),
    Anonymous = require("../tree/anonymous.js"),
    URL = require("../tree/url.js"),
    Operation = require("../tree/operation.js"),
    functionRegistry = require("./function-registry.js");

var isa = function (n, Type) {
    return (n instanceof Type) ? Keyword.True : Keyword.False;
    },
    isunit = function (n, unit) {
        return (n instanceof Dimension) && n.unit.is(unit.value || unit) ? Keyword.True : Keyword.False;
    };
functionRegistry.addMultiple({
    iscolor: function (n) {
        return isa(n, Color);
    },
    isnumber: function (n) {
        return isa(n, Dimension);
    },
    isstring: function (n) {
        return isa(n, Quoted);
    },
    iskeyword: function (n) {
        return isa(n, Keyword);
    },
    isurl: function (n) {
        return isa(n, URL);
    },
    ispixel: function (n) {
        return isunit(n, 'px');
    },
    ispercentage: function (n) {
        return isunit(n, '%');
    },
    isem: function (n) {
        return isunit(n, 'em');
    },
    isunit: isunit,
    unit: function (val, unit) {
        if(!(val instanceof Dimension)) {
            throw { type: "Argument", message: "the first argument to unit must be a number" + (val instanceof Operation ? ". Have you forgotten parenthesis?" : "") };
        }
        if (unit) {
            if (unit instanceof Keyword) {
                unit = unit.value;
            } else {
                unit = unit.toCSS();
            }
        } else {
            unit = "";
        }
        return new(Dimension)(val.value, unit);
    },
    "get-unit": function (n) {
        return new(Anonymous)(n.unit);
    },
    extract: function(values, index) {
        index = index.value - 1; // (1-based index)
        // handle non-array values as an array of length 1
        // return 'undefined' if index is invalid
        return Array.isArray(values.value) ?
            values.value[index] : Array(values)[index];
    },
    length: function(values) {
        var n = Array.isArray(values.value) ? values.value.length : 1;
        return new Dimension(n);
    }
});

},{"../tree/anonymous.js":27,"../tree/color.js":31,"../tree/dimension.js":37,"../tree/keyword.js":46,"../tree/operation.js":52,"../tree/quoted.js":54,"../tree/url.js":61,"./function-registry.js":12}],19:[function(require,module,exports){

var LessError = module.exports = function LessError(parser, e, env) {
    var input = parser.getInput(e, env),
        loc = parser.getLocation(e.index, input),
        line = loc.line,
        col  = loc.column,
        callLine = e.call && parser.getLocation(e.call, input).line,
        lines = input.split('\n');

    this.type = e.type || 'Syntax';
    this.message = e.message;
    this.filename = e.filename || env.currentFileInfo.filename;
    this.index = e.index;
    this.line = typeof(line) === 'number' ? line + 1 : null;
    this.callLine = callLine + 1;
    this.callExtract = lines[callLine];
    this.stack = e.stack;
    this.column = col;
    this.extract = [
        lines[line - 1],
        lines[line],
        lines[line + 1]
    ];
};

LessError.prototype = new Error();
LessError.prototype.constructor = LessError;

},{}],20:[function(require,module,exports){
module.exports = function(environment) {
    var less = {
        version: [2, 0, 0],
        data: require('./data/index.js'),
        tree: require('./tree/index.js'),
        visitor: require('./visitor/index.js'),
        Parser: require('./parser/parser.js')(environment),
        functions: require('./functions/index.js')(environment),
        contexts: require("./contexts.js"),
        environment: environment
    };

    return less;
};

},{"./contexts.js":2,"./data/index.js":4,"./functions/index.js":13,"./parser/parser.js":24,"./tree/index.js":43,"./visitor/index.js":66}],21:[function(require,module,exports){
// Split the input into chunks.
module.exports = function (input, fail) {
    var len = input.length, level = 0, parenLevel = 0,
        lastOpening, lastOpeningParen, lastMultiComment, lastMultiCommentEndBrace,
        chunks = [], emitFrom = 0,
        chunkerCurrentIndex, currentChunkStartIndex, cc, cc2, matched;

    function emitChunk(force) {
        var len = chunkerCurrentIndex - emitFrom;
        if (((len < 512) && !force) || !len) {
            return;
        }
        chunks.push(input.slice(emitFrom, chunkerCurrentIndex + 1));
        emitFrom = chunkerCurrentIndex + 1;
    }

    for (chunkerCurrentIndex = 0; chunkerCurrentIndex < len; chunkerCurrentIndex++) {
        cc = input.charCodeAt(chunkerCurrentIndex);
        if (((cc >= 97) && (cc <= 122)) || (cc < 34)) {
            // a-z or whitespace
            continue;
        }

        switch (cc) {
            case 40:                        // (
                parenLevel++;
                lastOpeningParen = chunkerCurrentIndex;
                continue;
            case 41:                        // )
                if (--parenLevel < 0) {
                    return fail("missing opening `(`", chunkerCurrentIndex);
                }
                continue;
            case 59:                        // ;
                if (!parenLevel) { emitChunk(); }
                continue;
            case 123:                       // {
                level++;
                lastOpening = chunkerCurrentIndex;
                continue;
            case 125:                       // }
                if (--level < 0) {
                    return fail("missing opening `{`", chunkerCurrentIndex);
                }
                if (!level && !parenLevel) { emitChunk(); }
                continue;
            case 92:                        // \
                if (chunkerCurrentIndex < len - 1) { chunkerCurrentIndex++; continue; }
                return fail("unescaped `\\`", chunkerCurrentIndex);
            case 34:
            case 39:
            case 96:                        // ", ' and `
                matched = 0;
                currentChunkStartIndex = chunkerCurrentIndex;
                for (chunkerCurrentIndex = chunkerCurrentIndex + 1; chunkerCurrentIndex < len; chunkerCurrentIndex++) {
                    cc2 = input.charCodeAt(chunkerCurrentIndex);
                    if (cc2 > 96) { continue; }
                    if (cc2 == cc) { matched = 1; break; }
                    if (cc2 == 92) {        // \
                        if (chunkerCurrentIndex == len - 1) {
                            return fail("unescaped `\\`", chunkerCurrentIndex);
                        }
                        chunkerCurrentIndex++;
                    }
                }
                if (matched) { continue; }
                return fail("unmatched `" + String.fromCharCode(cc) + "`", currentChunkStartIndex);
            case 47:                        // /, check for comment
                if (parenLevel || (chunkerCurrentIndex == len - 1)) { continue; }
                cc2 = input.charCodeAt(chunkerCurrentIndex + 1);
                if (cc2 == 47) {
                    // //, find lnfeed
                    for (chunkerCurrentIndex = chunkerCurrentIndex + 2; chunkerCurrentIndex < len; chunkerCurrentIndex++) {
                        cc2 = input.charCodeAt(chunkerCurrentIndex);
                        if ((cc2 <= 13) && ((cc2 == 10) || (cc2 == 13))) { break; }
                    }
                } else if (cc2 == 42) {
                    // /*, find */
                    lastMultiComment = currentChunkStartIndex = chunkerCurrentIndex;
                    for (chunkerCurrentIndex = chunkerCurrentIndex + 2; chunkerCurrentIndex < len - 1; chunkerCurrentIndex++) {
                        cc2 = input.charCodeAt(chunkerCurrentIndex);
                        if (cc2 == 125) { lastMultiCommentEndBrace = chunkerCurrentIndex; }
                        if (cc2 != 42) { continue; }
                        if (input.charCodeAt(chunkerCurrentIndex + 1) == 47) { break; }
                    }
                    if (chunkerCurrentIndex == len - 1) {
                        return fail("missing closing `*/`", currentChunkStartIndex);
                    }
                    chunkerCurrentIndex++;
                }
                continue;
            case 42:                       // *, check for unmatched */
                if ((chunkerCurrentIndex < len - 1) && (input.charCodeAt(chunkerCurrentIndex + 1) == 47)) {
                    return fail("unmatched `/*`", chunkerCurrentIndex);
                }
                continue;
        }
    }

    if (level !== 0) {
        if ((lastMultiComment > lastOpening) && (lastMultiCommentEndBrace > lastMultiComment)) {
            return fail("missing closing `}` or `*/`", lastOpening);
        } else {
            return fail("missing closing `}`", lastOpening);
        }
    } else if (parenLevel !== 0) {
        return fail("missing closing `)`", lastOpeningParen);
    }

    emitChunk(true);
    return chunks;
};

},{}],22:[function(require,module,exports){
var contexts = require("../contexts.js");
module.exports = function(environment, env, Parser) {
    var rootFilename = env && env.filename;
    return {
        paths: env.paths || [],  // Search paths, when importing
            queue: [],               // Files which haven't been imported yet
        files: env.files,        // Holds the imported parse trees
        contents: env.contents,  // Holds the imported file contents
        contentsIgnoredChars: env.contentsIgnoredChars, // lines inserted, not in the original less
        mime:  env.mime,         // MIME type of .less files
        error: null,             // Error in parsing/evaluating an import
        push: function (path, currentFileInfo, importOptions, callback) {
            var parserImports = this;
            this.queue.push(path);

            var fileParsedFunc = function (e, root, fullPath) {
                parserImports.queue.splice(parserImports.queue.indexOf(path), 1); // Remove the path from the queue

                var importedPreviously = fullPath === rootFilename;

                parserImports.files[fullPath] = root;                        // Store the root

                if (e && !parserImports.error) { parserImports.error = e; }

                callback(e, root, importedPreviously, fullPath);
            };

            var newFileInfo = {
                relativeUrls: env.relativeUrls,
                entryPath: currentFileInfo.entryPath,
                rootpath: currentFileInfo.rootpath,
                rootFilename: currentFileInfo.rootFilename
            };

            environment.loadFile(env, path, currentFileInfo.currentDirectory, function loadFileCallback(e, contents, resolvedFilename) {
                if (e) {
                    fileParsedFunc(e);
                    return;
                }

                // Pass on an updated rootpath if path of imported file is relative and file
                // is in a (sub|sup) directory
                //
                // Examples:
                // - If path of imported file is 'module/nav/nav.less' and rootpath is 'less/',
                //   then rootpath should become 'less/module/nav/'
                // - If path of imported file is '../mixins.less' and rootpath is 'less/',
                //   then rootpath should become 'less/../'
                newFileInfo.currentDirectory = environment.getPath(env, resolvedFilename);
                if(newFileInfo.relativeUrls) {
                    newFileInfo.rootpath = environment.join((env.rootpath || ""), environment.pathDiff(newFileInfo.currentDirectory, newFileInfo.entryPath));
                    if (!environment.isPathAbsolute(env, newFileInfo.rootpath) && environment.alwaysMakePathsAbsolute()) {
                        newFileInfo.rootpath = environment.join(newFileInfo.entryPath, newFileInfo.rootpath);
                    }
                }
                newFileInfo.filename = resolvedFilename;

                var newEnv = new contexts.parseEnv(env);

                newEnv.currentFileInfo = newFileInfo;
                newEnv.processImports = false;
                newEnv.contents[resolvedFilename] = contents;

                if (currentFileInfo.reference || importOptions.reference) {
                    newFileInfo.reference = true;
                }

                if (importOptions.inline) {
                    fileParsedFunc(null, contents, resolvedFilename);
                } else {
                    new(Parser)(newEnv).parse(contents, function (e, root) {
                        fileParsedFunc(e, root, resolvedFilename);
                    });
                }
            });
        }
    };
};

},{"../contexts.js":2}],23:[function(require,module,exports){
var chunker = require('./chunker.js'),
    LessError = require('../less-error.js');
module.exports = function() {
    var input,       // LeSS input string
        j,           // current chunk
        saveStack = [],   // holds state for backtracking
        furthest,    // furthest index the parser has gone to
        furthestPossibleErrorMessage,// if this is furthest we got to, this is the probably cause
        chunks,      // chunkified input
        current,     // current chunk
        currentPos,  // index of current chunk, in `input`
        parserInput = {};

    parserInput.save = function() {
        currentPos = parserInput.i;
        saveStack.push( { current: current, i: parserInput.i, j: j });
    };
    parserInput.restore = function(possibleErrorMessage) {
        if (parserInput.i > furthest) {
            furthest = parserInput.i;
            furthestPossibleErrorMessage = possibleErrorMessage;
        }
        var state = saveStack.pop();
        current = state.current;
        currentPos = parserInput.i = state.i;
        j = state.j;
    };
    parserInput.forget = function() {
        saveStack.pop();
    };
    function sync() {
        if (parserInput.i > currentPos) {
            current = current.slice(parserInput.i - currentPos);
            currentPos = parserInput.i;
        }
    }
    parserInput.isWhitespace = function (offset) {
        var pos = parserInput.i + (offset || 0),
            code = input.charCodeAt(pos);
        return (code === CHARCODE_SPACE || code === CHARCODE_CR || code === CHARCODE_TAB || code === CHARCODE_LF);
    };
    //
    // Parse from a token, regexp or string, and move forward if match
    //
    parserInput.$ = function(tok) {
        var tokType = typeof tok,
            match, length;

        // Either match a single character in the input,
        // or match a regexp in the current chunk (`current`).
        //
        if (tokType === "string") {
            if (input.charAt(parserInput.i) !== tok) {
                return null;
            }
            skipWhitespace(1);
            return tok;
        }

        // regexp
        sync();
        if (! (match = tok.exec(current))) {
            return null;
        }

        length = match[0].length;

        // The match is confirmed, add the match length to `i`,
        // and consume any extra white-space characters (' ' || '\n')
        // which come after that. The reason for this is that LeSS's
        // grammar is mostly white-space insensitive.
        //
        skipWhitespace(length);

        if(typeof(match) === 'string') {
            return match;
        } else {
            return match.length === 1 ? match[0] : match;
        }
    };

    // Specialization of $(tok)
    parserInput.$re = function(tok) {
        if (parserInput.i > currentPos) {
            current = current.slice(parserInput.i - currentPos);
            currentPos = parserInput.i;
        }
        var m = tok.exec(current);
        if (!m) {
            return null;
        }

        skipWhitespace(m[0].length);
        if(typeof m === "string") {
            return m;
        }

        return m.length === 1 ? m[0] : m;
    };

    // Specialization of $(tok)
    parserInput.$char = function(tok) {
        if (input.charAt(parserInput.i) !== tok) {
            return null;
        }
        skipWhitespace(1);
        return tok;
    };

    var CHARCODE_SPACE = 32,
        CHARCODE_TAB = 9,
        CHARCODE_LF = 10,
        CHARCODE_CR = 13,
        CHARCODE_PLUS = 43,
        CHARCODE_COMMA = 44,
        CHARCODE_FORWARD_SLASH = 47,
        CHARCODE_9 = 57;

    parserInput.autoCommentAbsorb = true;
    parserInput.commentStore = [];
    parserInput.finished = false;

    var skipWhitespace = function(length) {
        var oldi = parserInput.i, oldj = j,
            curr = parserInput.i - currentPos,
            endIndex = parserInput.i + current.length - curr,
            mem = (parserInput.i += length),
            inp = input,
            c, nextChar, comment;

        for (; parserInput.i < endIndex; parserInput.i++) {
            c = inp.charCodeAt(parserInput.i);

            if (parserInput.autoCommentAbsorb && c === CHARCODE_FORWARD_SLASH) {
                nextChar = inp[parserInput.i + 1];
                if (nextChar === '/') {
                    comment = {index: parserInput.i, isLineComment: true};
                    var nextNewLine = inp.indexOf("\n", parserInput.i + 1);
                    if (nextNewLine < 0) {
                        nextNewLine = endIndex;
                    }
                    parserInput.i = nextNewLine;
                    comment.text = inp.substr(comment.i, parserInput.i - comment.i);
                    parserInput.commentStore.push(comment);
                    continue;
                } else if (nextChar === '*') {
                    var haystack = inp.substr(parserInput.i);
                    var comment_search_result = haystack.match(/^\/\*(?:[^*]|\*+[^\/*])*\*+\//);
                    if (comment_search_result) {
                        comment = {
                            index: parserInput.i,
                            text: comment_search_result[0],
                            isLineComment: false
                        };
                        parserInput.i += comment.text.length - 1;
                        parserInput.commentStore.push(comment);
                        continue;
                    }
                }
                break;
            }

            if ((c !== CHARCODE_SPACE) && (c !== CHARCODE_LF) && (c !== CHARCODE_TAB) && (c !== CHARCODE_CR)) {
                break;
            }
        }

        current = current.slice(length + parserInput.i - mem + curr);
        currentPos = parserInput.i;

        if (!current.length) {
            if (j < chunks.length - 1)
            {
                current = chunks[++j];
                skipWhitespace(0); // skip space at the beginning of a chunk
                return true; // things changed
            }
            parserInput.finished = true;
        }

        return oldi !== parserInput.i || oldj !== j;
    };

    // Same as $(), but don't change the state of the parser,
    // just return the match.
    parserInput.peek = function(tok) {
        if (typeof(tok) === 'string') {
            return input.charAt(parserInput.i) === tok;
        } else {
            return tok.test(current);
        }
    };

    // Specialization of peek()
    // TODO remove or change some currentChar calls to peekChar
    parserInput.peekChar = function(tok) {
        return input.charAt(parserInput.i) === tok;
    };

    parserInput.currentChar = function() {
        return input.charAt(parserInput.i);
    };

    parserInput.getInput = function() {
        return input;
    };

    parserInput.peekNotNumeric = function() {
        var c = input.charCodeAt(parserInput.i);
        //Is the first char of the dimension 0-9, '.', '+' or '-'
        return (c > CHARCODE_9 || c < CHARCODE_PLUS) || c === CHARCODE_FORWARD_SLASH || c === CHARCODE_COMMA;
    };

    parserInput.getLocation = function(index, inputStream) {
        inputStream = inputStream == null ? input : inputStream;

        var n = index + 1,
            line = null,
            column = -1;

        while (--n >= 0 && inputStream.charAt(n) !== '\n') {
            column++;
        }

        if (typeof index === 'number') {
            line = (inputStream.slice(0, index).match(/\n/g) || "").length;
        }

        return {
            line: line,
            column: column
        };
    };

    parserInput.start = function(str, chunkInput, parser, env) {
        input = str;
        parserInput.i = j = currentPos = furthest = 0;

        // chunking apparantly makes things quicker (but my tests indicate
        // it might actually make things slower in node at least)
        // and it is a non-perfect parse - it can't recognise
        // unquoted urls, meaning it can't distinguish comments
        // meaning comments with quotes or {}() in them get 'counted'
        // and then lead to parse errors.
        // In addition if the chunking chunks in the wrong place we might
        // not be able to parse a parser statement in one go
        // this is officially deprecated but can be switched on via an option
        // in the case it causes too much performance issues.
        if (chunkInput) {
            chunks = chunker(str, function fail(msg, index) {
                throw new(LessError)(parser, {
                    index: index,
                    type: 'Parse',
                    message: msg,
                    filename: env.currentFileInfo.filename
                }, env);
            });
        } else {
            chunks = [str];
        }

        current = chunks[0];

        skipWhitespace(0);
    };

    parserInput.end = function() {
        var message,
            isFinished = parserInput.i >= input.length - 1;

        if (parserInput.i < furthest) {
            message = furthestPossibleErrorMessage;
            parserInput.i = furthest;
        }
        return {
            isFinished: isFinished,
            furthest: parserInput.i,
            furthestPossibleErrorMessage: message,
            furthestReachedEnd: parserInput.i >= input.length - 1,
            furthestChar: input[parserInput.i]
        };
    };

    return parserInput;
};

},{"../less-error.js":19,"./chunker.js":21}],24:[function(require,module,exports){
var LessError = require('../less-error.js'),
    tree = require("../tree/index.js"),
    visitor = require("../visitor/index.js"),
    contexts = require("../contexts.js"),
    getImportManager = require("./imports.js"),
    getParserInput = require("./parser-input.js");

module.exports = function(environment) {
var SourceMapOutput = require("../source-map-output")(environment);
//
// less.js - parser
//
//    A relatively straight-forward predictive parser.
//    There is no tokenization/lexing stage, the input is parsed
//    in one sweep.
//
//    To make the parser fast enough to run in the browser, several
//    optimization had to be made:
//
//    - Matching and slicing on a huge input is often cause of slowdowns.
//      The solution is to chunkify the input into smaller strings.
//      The chunks are stored in the `chunks` var,
//      `j` holds the current chunk index, and `currentPos` holds
//      the index of the current chunk in relation to `input`.
//      This gives us an almost 4x speed-up.
//
//    - In many cases, we don't need to match individual tokens;
//      for example, if a value doesn't hold any variables, operations
//      or dynamic references, the parser can effectively 'skip' it,
//      treating it as a literal.
//      An example would be '1px solid #000' - which evaluates to itself,
//      we don't need to know what the individual components are.
//      The drawback, of course is that you don't get the benefits of
//      syntax-checking on the CSS. This gives us a 50% speed-up in the parser,
//      and a smaller speed-up in the code-gen.
//
//
//    Token matching is done with the `$` function, which either takes
//    a terminal string or regexp, or a non-terminal function to call.
//    It also takes care of moving all the indices forwards.
//
//
var Parser = function Parser(env) {
    var parser,
        parsers,
        parserInput = getParserInput();

    // Top parser on an import tree must be sure there is one "env"
    // which will then be passed around by reference.
    if (!(env instanceof contexts.parseEnv)) {
        env = new contexts.parseEnv(env);
    }
    this.env = env;

    var imports = this.imports = getImportManager(environment, env, Parser);

    function expect(arg, msg, index) {
        // some older browsers return typeof 'function' for RegExp
        var result = (Object.prototype.toString.call(arg) === '[object Function]') ? arg.call(parsers) : parserInput.$(arg);
        if (result) {
            return result;
        }
        error(msg || (typeof(arg) === 'string' ? "expected '" + arg + "' got '" + parserInput.currentChar() + "'"
                                               : "unexpected token"));
    }

    // Specialization of expect()
    function expectChar(arg, msg) {
        if (parserInput.$char(arg)) {
            return arg;
        }
        error(msg || "expected '" + arg + "' got '" + parserInput.currentChar() + "'");
    }

    function error(msg, type) {
        var e = new Error(msg);
        e.index = parserInput.i;
        e.type = type || 'Syntax';
        throw e;
    }

    function getInput(e, env) {
        if (e.filename && env.currentFileInfo.filename && (e.filename !== env.currentFileInfo.filename)) {
            return parser.imports.contents[e.filename];
        } else {
            return parserInput.getInput();
        }
    }

    function getDebugInfo(index) {
        var filename = env.currentFileInfo.filename;
        filename = environment.getAbsolutePath(env, filename);

        return {
            lineNumber: parserInput.getLocation(index).line + 1,
            fileName: filename
        };
    }

    //
    // The Parser
    //
    parser = {

        imports: imports,
        //
        // Parse an input string into an abstract syntax tree,
        // @param str A string containing 'less' markup
        // @param callback call `callback` when done.
        // @param [additionalData] An optional map which can contains vars - a map (key, value) of variables to apply
        //
        parse: function (str, callback, additionalData) {
            var root, error = null, globalVars, modifyVars, preText = "";

            globalVars = (additionalData && additionalData.globalVars) ? Parser.serializeVars(additionalData.globalVars) + '\n' : '';
            modifyVars = (additionalData && additionalData.modifyVars) ? '\n' + Parser.serializeVars(additionalData.modifyVars) : '';

            if (globalVars || (additionalData && additionalData.banner)) {
                preText = ((additionalData && additionalData.banner) ? additionalData.banner : "") + globalVars;
                parser.imports.contentsIgnoredChars[env.currentFileInfo.filename] = preText.length;
            }

            str = str.replace(/\r\n/g, '\n');
            // Remove potential UTF Byte Order Mark
            str = preText + str.replace(/^\uFEFF/, '') + modifyVars;
            parser.imports.contents[env.currentFileInfo.filename] = str;

            // Start with the primary rule.
            // The whole syntax tree is held under a Ruleset node,
            // with the `root` property set to true, so no `{}` are
            // output. The callback is called when the input is parsed.
            try {
                parserInput.start(str, env.chunkInput, parser, env);

                root = new(tree.Ruleset)(null, this.parsers.primary());
                root.root = true;
                root.firstRoot = true;
            } catch (e) {
                return callback(new LessError(parser, e, env));
            }

            root.toCSS = (function (evaluate) {
                return function (options, variables) {
                    options = options || {};
                    var evaldRoot,
                        css,
                        evalEnv = new contexts.evalEnv(options);

                    //
                    // Allows setting variables with a hash, so:
                    //
                    //   `{ color: new(tree.Color)('#f01') }` will become:
                    //
                    //   new(tree.Rule)('@color',
                    //     new(tree.Value)([
                    //       new(tree.Expression)([
                    //         new(tree.Color)('#f01')
                    //       ])
                    //     ])
                    //   )
                    //
                    if (typeof(variables) === 'object' && !Array.isArray(variables)) {
                        variables = Object.keys(variables).map(function (k) {
                            var value = variables[k];

                            if (! (value instanceof tree.Value)) {
                                if (! (value instanceof tree.Expression)) {
                                    value = new(tree.Expression)([value]);
                                }
                                value = new(tree.Value)([value]);
                            }
                            return new(tree.Rule)('@' + k, value, false, null, 0);
                        });
                        evalEnv.frames = [new(tree.Ruleset)(null, variables)];
                    }

                    try {
                        var preEvalVisitors = [],
                            visitors = [
                                new(visitor.JoinSelectorVisitor)(),
                                new(visitor.ExtendVisitor)(),
                                new(visitor.ToCSSVisitor)({compress: Boolean(options.compress)})
                            ], i, root = this;

                        if (options.plugins) {
                            for(i =0; i < options.plugins.length; i++) {
                                if (options.plugins[i].isPreEvalVisitor) {
                                    preEvalVisitors.push(options.plugins[i]);
                                } else {
                                    if (options.plugins[i].isPreVisitor) {
                                        visitors.splice(0, 0, options.plugins[i]);
                                    } else {
                                        visitors.push(options.plugins[i]);
                                    }
                                }
                            }
                        }

                        for(i = 0; i < preEvalVisitors.length; i++) {
                            preEvalVisitors[i].run(root);
                        }

                        evaldRoot = evaluate.call(root, evalEnv);

                        for(i = 0; i < visitors.length; i++) {
                            visitors[i].run(evaldRoot);
                        }

                        if (options.sourceMap) {
                            evaldRoot = new SourceMapOutput(
                                {
                                    contentsIgnoredCharsMap: parser.imports.contentsIgnoredChars,
                                    writeSourceMap: options.writeSourceMap,
                                    rootNode: evaldRoot,
                                    contentsMap: parser.imports.contents,
                                    sourceMapFilename: options.sourceMapFilename,
                                    sourceMapURL: options.sourceMapURL,
                                    outputFilename: options.sourceMapOutputFilename,
                                    sourceMapBasepath: options.sourceMapBasepath,
                                    sourceMapRootpath: options.sourceMapRootpath,
                                    outputSourceFiles: options.outputSourceFiles,
                                    sourceMapGenerator: options.sourceMapGenerator
                                });
                        }

                        css = evaldRoot.toCSS({
                                compress: Boolean(options.compress),
                                dumpLineNumbers: env.dumpLineNumbers,
                                strictUnits: Boolean(options.strictUnits),
                                numPrecision: 8});
                    } catch (e) {
                        throw new LessError(parser, e, env);
                    }

                    var CleanCSS = environment.getCleanCSS();
                    if (options.cleancss && CleanCSS) {
                        var cleancssOptions = options.cleancssOptions || {};

                        if (cleancssOptions.keepSpecialComments === undefined) {
                            cleancssOptions.keepSpecialComments = "*";
                        }
                        cleancssOptions.processImport = false;
                        cleancssOptions.noRebase = true;
                        if (cleancssOptions.noAdvanced === undefined) {
                            cleancssOptions.noAdvanced = true;
                        }

                        return new CleanCSS(cleancssOptions).minify(css);
                    } else if (options.compress) {
                        return css.replace(/(^(\s)+)|((\s)+$)/g, "");
                    } else {
                        return css;
                    }
                };
            })(root.eval);

            // If `i` is smaller than the `input.length - 1`,
            // it means the parser wasn't able to parse the whole
            // string, so we've got a parsing error.
            //
            // We try to extract a \n delimited string,
            // showing the line where the parse error occurred.
            // We split it up into two parts (the part which parsed,
            // and the part which didn't), so we can color them differently.
            var endInfo = parserInput.end();
            if (!endInfo.isFinished) {

                var message = endInfo.furthestPossibleErrorMessage;

                if (!message) {
                    message = "Unrecognised input";
                    if (endInfo.furthestChar === '}') {
                        message += ". Possibly missing opening '{'";
                    } else if (endInfo.furthestChar === ')') {
                        message += ". Possibly missing opening '('";
                    } else if (endInfo.furthestReachedEnd) {
                        message += ". Possibly missing something";
                    }
                }

                error = new LessError(parser, {
                    type: "Parse",
                    message: message,
                    index: endInfo.furthest,
                    filename: env.currentFileInfo.filename
                }, env);
            }

            var finish = function (e) {
                e = error || e || parser.imports.error;

                if (e) {
                    if (!(e instanceof LessError)) {
                        e = new LessError(parser, e, env);
                    }

                    return callback(e);
                }
                else {
                    return callback(null, root);
                }
            };

            if (env.processImports !== false) {
                new visitor.ImportVisitor(this.imports, finish)
                    .run(root);
            } else {
                return finish();
            }
        },

        //
        // Here in, the parsing rules/functions
        //
        // The basic structure of the syntax tree generated is as follows:
        //
        //   Ruleset ->  Rule -> Value -> Expression -> Entity
        //
        // Here's some Less code:
        //
        //    .class {
        //      color: #fff;
        //      border: 1px solid #000;
        //      width: @w + 4px;
        //      > .child {...}
        //    }
        //
        // And here's what the parse tree might look like:
        //
        //     Ruleset (Selector '.class', [
        //         Rule ("color",  Value ([Expression [Color #fff]]))
        //         Rule ("border", Value ([Expression [Dimension 1px][Keyword "solid"][Color #000]]))
        //         Rule ("width",  Value ([Expression [Operation "+" [Variable "@w"][Dimension 4px]]]))
        //         Ruleset (Selector [Element '>', '.child'], [...])
        //     ])
        //
        //  In general, most rules will try to parse a token with the `$()` function, and if the return
        //  value is truly, will return a new node, of the relevant type. Sometimes, we need to check
        //  first, before parsing, that's when we use `peek()`.
        //
        parsers: parsers = {
            //
            // The `primary` rule is the *entry* and *exit* point of the parser.
            // The rules here can appear at any level of the parse tree.
            //
            // The recursive nature of the grammar is an interplay between the `block`
            // rule, which represents `{ ... }`, the `ruleset` rule, and this `primary` rule,
            // as represented by this simplified grammar:
            //
            //     primary  →  (ruleset | rule)+
            //     ruleset  →  selector+ block
            //     block    →  '{' primary '}'
            //
            // Only at one point is the primary rule not called from the
            // block rule: at the root level.
            //
            primary: function () {
                var mixin = this.mixin, root = [], node;

                while (!parserInput.finished)
                {
                    while(true) {
                        node = this.comment();
                        if (!node) { break; }
                        root.push(node);
                    }
                    if (parserInput.peek('}')) {
                        break;
                    }
                    node = this.extendRule() || mixin.definition() || this.rule() || this.ruleset() ||
                        mixin.call() || this.rulesetCall() || this.directive();
                    if (node) {
                        root.push(node);
                    } else {
                        if (!(parserInput.$re(/^[\s\n]+/) || parserInput.$re(/^;+/))) {
                            break;
                        }
                    }
                }

                return root;
            },

            // comments are collected by the main parsing mechanism and then assigned to nodes
            // where the current structure allows it
            comment: function () {
                if (parserInput.commentStore.length) {
                    var comment = parserInput.commentStore.shift();
                    return new(tree.Comment)(comment.text, comment.isLineComment, comment.index, env.currentFileInfo);
                }
            },

            //
            // Entities are tokens which can be found inside an Expression
            //
            entities: {
                //
                // A string, which supports escaping " and '
                //
                //     "milky way" 'he\'s the one!'
                //
                quoted: function () {
                    var str, index = parserInput.i;

                    str = parserInput.$re(/^(~)?("((?:[^"\\\r\n]|\\.)*)"|'((?:[^'\\\r\n]|\\.)*)')/);
                    if (str) {
                        return new(tree.Quoted)(str[2], str[3] || str[4], Boolean(str[1]), index, env.currentFileInfo);
                    }
                },

                //
                // A catch-all word, such as:
                //
                //     black border-collapse
                //
                keyword: function () {
                    var k = parserInput.$re(/^%|^[_A-Za-z-][_A-Za-z0-9-]*/);
                    if (k) {
                        return tree.Color.fromKeyword(k) || new(tree.Keyword)(k);
                    }
                },

                //
                // A function call
                //
                //     rgb(255, 0, 255)
                //
                // We also try to catch IE's `alpha()`, but let the `alpha` parser
                // deal with the details.
                //
                // The arguments are parsed with the `entities.arguments` parser.
                //
                call: function () {
                    var name, nameLC, args, alpha, index = parserInput.i;

                    if (parserInput.peek(/^url\(/i)) {
                        return;
                    }

                    parserInput.save();

                    name = parserInput.$re(/^([\w-]+|%|progid:[\w\.]+)\(/);
                    if (!name) { parserInput.forget(); return; }

                    name = name[1];
                    nameLC = name.toLowerCase();

                    if (nameLC === 'alpha') {
                        alpha = parsers.alpha();
                        if(alpha) {
                            return alpha;
                        }
                    }

                    args = this.arguments();

                    if (! parserInput.$char(')')) {
                        parserInput.restore("Could not parse call arguments or missing ')'");
                        return;
                    }

                    parserInput.forget();
                    return new(tree.Call)(name, args, index, env.currentFileInfo);
                },
                arguments: function () {
                    var args = [], arg;

                    while (true) {
                        arg = this.assignment() || parsers.expression();
                        if (!arg) {
                            break;
                        }
                        args.push(arg);
                        if (! parserInput.$char(',')) {
                            break;
                        }
                    }
                    return args;
                },
                literal: function () {
                    return this.dimension() ||
                           this.color() ||
                           this.quoted() ||
                           this.unicodeDescriptor();
                },

                // Assignments are argument entities for calls.
                // They are present in ie filter properties as shown below.
                //
                //     filter: progid:DXImageTransform.Microsoft.Alpha( *opacity=50* )
                //

                assignment: function () {
                    var key, value;
                    key = parserInput.$re(/^\w+(?=\s?=)/i);
                    if (!key) {
                        return;
                    }
                    if (!parserInput.$char('=')) {
                        return;
                    }
                    value = parsers.entity();
                    if (value) {
                        return new(tree.Assignment)(key, value);
                    }
                },

                //
                // Parse url() tokens
                //
                // We use a specific rule for urls, because they don't really behave like
                // standard function calls. The difference is that the argument doesn't have
                // to be enclosed within a string, so it can't be parsed as an Expression.
                //
                url: function () {
                    var value;

                    if (parserInput.currentChar() !== 'u' || !parserInput.$re(/^url\(/)) {
                        return;
                    }

                    parserInput.autoCommentAbsorb = false;

                    value = this.quoted() || this.variable() ||
                            parserInput.$re(/^(?:(?:\\[\(\)'"])|[^\(\)'"])+/) || "";

                    parserInput.autoCommentAbsorb = true;

                    expectChar(')');

                    return new(tree.URL)((value.value != null || value instanceof tree.Variable)
                                        ? value : new(tree.Anonymous)(value), env.currentFileInfo);
                },

                //
                // A Variable entity, such as `@fink`, in
                //
                //     width: @fink + 2px
                //
                // We use a different parser for variable definitions,
                // see `parsers.variable`.
                //
                variable: function () {
                    var name, index = parserInput.i;

                    if (parserInput.currentChar() === '@' && (name = parserInput.$re(/^@@?[\w-]+/))) {
                        return new(tree.Variable)(name, index, env.currentFileInfo);
                    }
                },

                // A variable entity useing the protective {} e.g. @{var}
                variableCurly: function () {
                    var curly, index = parserInput.i;

                    if (parserInput.currentChar() === '@' && (curly = parserInput.$re(/^@\{([\w-]+)\}/))) {
                        return new(tree.Variable)("@" + curly[1], index, env.currentFileInfo);
                    }
                },

                //
                // A Hexadecimal color
                //
                //     #4F3C2F
                //
                // `rgb` and `hsl` colors are parsed through the `entities.call` parser.
                //
                color: function () {
                    var rgb;

                    if (parserInput.currentChar() === '#' && (rgb = parserInput.$re(/^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})/))) {
                        var colorCandidateString = rgb.input.match(/^#([\w]+).*/); // strip colons, brackets, whitespaces and other characters that should not definitely be part of color string
                        colorCandidateString = colorCandidateString[1];
                        if (!colorCandidateString.match(/^[A-Fa-f0-9]+$/)) { // verify if candidate consists only of allowed HEX characters
                            error("Invalid HEX color code");
                        }
                        return new(tree.Color)(rgb[1]);
                    }
                },

                //
                // A Dimension, that is, a number and a unit
                //
                //     0.5em 95%
                //
                dimension: function () {
                    if (parserInput.peekNotNumeric()) {
                        return;
                    }

                    var value = parserInput.$re(/^([+-]?\d*\.?\d+)(%|[a-z]+)?/);
                    if (value) {
                        return new(tree.Dimension)(value[1], value[2]);
                    }
                },

                //
                // A unicode descriptor, as is used in unicode-range
                //
                // U+0??  or U+00A1-00A9
                //
                unicodeDescriptor: function () {
                    var ud;

                    ud = parserInput.$re(/^U\+[0-9a-fA-F?]+(\-[0-9a-fA-F?]+)?/);
                    if (ud) {
                        return new(tree.UnicodeDescriptor)(ud[0]);
                    }
                },

                //
                // JavaScript code to be evaluated
                //
                //     `window.location.href`
                //
                javascript: function () {
                    var js, index = parserInput.i;

                    js = parserInput.$re(/^(~)?`([^`]*)`/);
                    if (js) {
                        return new(tree.JavaScript)(js[2], index, Boolean(js[1]));
                    }
                }
            },

            //
            // The variable part of a variable definition. Used in the `rule` parser
            //
            //     @fink:
            //
            variable: function () {
                var name;

                if (parserInput.currentChar() === '@' && (name = parserInput.$re(/^(@[\w-]+)\s*:/))) { return name[1]; }
            },

            //
            // The variable part of a variable definition. Used in the `rule` parser
            //
            //     @fink();
            //
            rulesetCall: function () {
                var name;

                if (parserInput.currentChar() === '@' && (name = parserInput.$re(/^(@[\w-]+)\s*\(\s*\)\s*;/))) {
                    return new tree.RulesetCall(name[1]);
                }
            },

            //
            // extend syntax - used to extend selectors
            //
            extend: function(isRule) {
                var elements, e, index = parserInput.i, option, extendList, extend;

                if (!(isRule ? parserInput.$re(/^&:extend\(/) : parserInput.$re(/^:extend\(/))) { return; }

                do {
                    option = null;
                    elements = null;
                    while (! (option = parserInput.$re(/^(all)(?=\s*(\)|,))/))) {
                        e = this.element();
                        if (!e) { break; }
                        if (elements) { elements.push(e); } else { elements = [ e ]; }
                    }

                    option = option && option[1];
                    if (!elements)
                        error("Missing target selector for :extend().");
                    extend = new(tree.Extend)(new(tree.Selector)(elements), option, index);
                    if (extendList) { extendList.push(extend); } else { extendList = [ extend ]; }

                } while(parserInput.$char(","));

                expect(/^\)/);

                if (isRule) {
                    expect(/^;/);
                }

                return extendList;
            },

            //
            // extendRule - used in a rule to extend all the parent selectors
            //
            extendRule: function() {
                return this.extend(true);
            },

            //
            // Mixins
            //
            mixin: {
                //
                // A Mixin call, with an optional argument list
                //
                //     #mixins > .square(#fff);
                //     .rounded(4px, black);
                //     .button;
                //
                // The `while` loop is there because mixins can be
                // namespaced, but we only support the child and descendant
                // selector for now.
                //
                call: function () {
                    var s = parserInput.currentChar(), important = false, index = parserInput.i, elemIndex,
                        elements, elem, e, c, args;

                    if (s !== '.' && s !== '#') { return; }

                    parserInput.save(); // stop us absorbing part of an invalid selector

                    while (true) {
                        elemIndex = parserInput.i;
                        e = parserInput.$re(/^[#.](?:[\w-]|\\(?:[A-Fa-f0-9]{1,6} ?|[^A-Fa-f0-9]))+/);
                        if (!e) {
                            break;
                        }
                        elem = new(tree.Element)(c, e, elemIndex, env.currentFileInfo);
                        if (elements) { elements.push(elem); } else { elements = [ elem ]; }
                        c = parserInput.$char('>');
                    }

                    if (elements) {
                        if (parserInput.$char('(')) {
                            args = this.args(true).args;
                            expectChar(')');
                        }

                        if (parsers.important()) {
                            important = true;
                        }

                        if (parsers.end()) {
                            parserInput.forget();
                            return new(tree.mixin.Call)(elements, args, index, env.currentFileInfo, important);
                        }
                    }

                    parserInput.restore();
                },
                args: function (isCall) {
                    var parsers = parser.parsers, entities = parsers.entities,
                        returner = { args:null, variadic: false },
                        expressions = [], argsSemiColon = [], argsComma = [],
                        isSemiColonSeperated, expressionContainsNamed, name, nameLoop, value, arg;

                    parserInput.save();

                    while (true) {
                        if (isCall) {
                            arg = parsers.detachedRuleset() || parsers.expression();
                        } else {
                            parserInput.commentStore.length = 0;
                            if (parserInput.currentChar() === '.' && parserInput.$re(/^\.{3}/)) {
                                returner.variadic = true;
                                if (parserInput.$char(";") && !isSemiColonSeperated) {
                                    isSemiColonSeperated = true;
                                }
                                (isSemiColonSeperated ? argsSemiColon : argsComma)
                                    .push({ variadic: true });
                                break;
                            }
                            arg = entities.variable() || entities.literal() || entities.keyword();
                        }

                        if (!arg) {
                            break;
                        }

                        nameLoop = null;
                        if (arg.throwAwayComments) {
                            arg.throwAwayComments();
                        }
                        value = arg;
                        var val = null;

                        if (isCall) {
                            // Variable
                            if (arg.value && arg.value.length == 1) {
                                val = arg.value[0];
                            }
                        } else {
                            val = arg;
                        }

                        if (val && val instanceof tree.Variable) {
                            if (parserInput.$char(':')) {
                                if (expressions.length > 0) {
                                    if (isSemiColonSeperated) {
                                        error("Cannot mix ; and , as delimiter types");
                                    }
                                    expressionContainsNamed = true;
                                }

                                // we do not support setting a ruleset as a default variable - it doesn't make sense
                                // However if we do want to add it, there is nothing blocking it, just don't error
                                // and remove isCall dependency below
                                value = (isCall && parsers.detachedRuleset()) || parsers.expression();

                                if (!value) {
                                    if (isCall) {
                                        error("could not understand value for named argument");
                                    } else {
                                        parserInput.restore();
                                        returner.args = [];
                                        return returner;
                                    }
                                }
                                nameLoop = (name = val.name);
                            } else if (!isCall && parserInput.$re(/^\.{3}/)) {
                                returner.variadic = true;
                                if (parserInput.$char(";") && !isSemiColonSeperated) {
                                    isSemiColonSeperated = true;
                                }
                                (isSemiColonSeperated ? argsSemiColon : argsComma)
                                    .push({ name: arg.name, variadic: true });
                                break;
                            } else if (!isCall) {
                                name = nameLoop = val.name;
                                value = null;
                            }
                        }

                        if (value) {
                            expressions.push(value);
                        }

                        argsComma.push({ name:nameLoop, value:value });

                        if (parserInput.$char(',')) {
                            continue;
                        }

                        if (parserInput.$char(';') || isSemiColonSeperated) {

                            if (expressionContainsNamed) {
                                error("Cannot mix ; and , as delimiter types");
                            }

                            isSemiColonSeperated = true;

                            if (expressions.length > 1) {
                                value = new(tree.Value)(expressions);
                            }
                            argsSemiColon.push({ name:name, value:value });

                            name = null;
                            expressions = [];
                            expressionContainsNamed = false;
                        }
                    }

                    parserInput.forget();
                    returner.args = isSemiColonSeperated ? argsSemiColon : argsComma;
                    return returner;
                },
                //
                // A Mixin definition, with a list of parameters
                //
                //     .rounded (@radius: 2px, @color) {
                //        ...
                //     }
                //
                // Until we have a finer grained state-machine, we have to
                // do a look-ahead, to make sure we don't have a mixin call.
                // See the `rule` function for more information.
                //
                // We start by matching `.rounded (`, and then proceed on to
                // the argument list, which has optional default values.
                // We store the parameters in `params`, with a `value` key,
                // if there is a value, such as in the case of `@radius`.
                //
                // Once we've got our params list, and a closing `)`, we parse
                // the `{...}` block.
                //
                definition: function () {
                    var name, params = [], match, ruleset, cond, variadic = false;
                    if ((parserInput.currentChar() !== '.' && parserInput.currentChar() !== '#') ||
                        parserInput.peek(/^[^{]*\}/)) {
                        return;
                    }

                    parserInput.save();

                    match = parserInput.$re(/^([#.](?:[\w-]|\\(?:[A-Fa-f0-9]{1,6} ?|[^A-Fa-f0-9]))+)\s*\(/);
                    if (match) {
                        name = match[1];

                        var argInfo = this.args(false);
                        params = argInfo.args;
                        variadic = argInfo.variadic;

                        // .mixincall("@{a}");
                        // looks a bit like a mixin definition..
                        // also
                        // .mixincall(@a: {rule: set;});
                        // so we have to be nice and restore
                        if (!parserInput.$char(')')) {
                            parserInput.restore("Missing closing ')'");
                            return;
                        }

                        parserInput.commentStore.length = 0;

                        if (parserInput.$re(/^when/)) { // Guard
                            cond = expect(parsers.conditions, 'expected condition');
                        }

                        ruleset = parsers.block();

                        if (ruleset) {
                            parserInput.forget();
                            return new(tree.mixin.Definition)(name, params, ruleset, cond, variadic);
                        } else {
                            parserInput.restore();
                        }
                    } else {
                        parserInput.forget();
                    }
                }
            },

            //
            // Entities are the smallest recognized token,
            // and can be found inside a rule's value.
            //
            entity: function () {
                var entities = this.entities;

                return this.comment() || entities.literal() || entities.variable() || entities.url() ||
                       entities.call()    || entities.keyword()  || entities.javascript();
            },

            //
            // A Rule terminator. Note that we use `peek()` to check for '}',
            // because the `block` rule will be expecting it, but we still need to make sure
            // it's there, if ';' was ommitted.
            //
            end: function () {
                return parserInput.$char(';') || parserInput.peek('}');
            },

            //
            // IE's alpha function
            //
            //     alpha(opacity=88)
            //
            alpha: function () {
                var value;

                if (! parserInput.$re(/^opacity=/i)) { return; }
                value = parserInput.$re(/^\d+/);
                if (!value) {
                    value = expect(this.entities.variable, "Could not parse alpha");
                }
                expectChar(')');
                return new(tree.Alpha)(value);
            },

            //
            // A Selector Element
            //
            //     div
            //     + h1
            //     #socks
            //     input[type="text"]
            //
            // Elements are the building blocks for Selectors,
            // they are made out of a `Combinator` (see combinator rule),
            // and an element name, such as a tag a class, or `*`.
            //
            element: function () {
                var e, c, v, index = parserInput.i;

                c = this.combinator();

                e = parserInput.$re(/^(?:\d+\.\d+|\d+)%/) || parserInput.$re(/^(?:[.#]?|:*)(?:[\w-]|[^\x00-\x9f]|\\(?:[A-Fa-f0-9]{1,6} ?|[^A-Fa-f0-9]))+/) ||
                    parserInput.$char('*') || parserInput.$char('&') || this.attribute() || parserInput.$re(/^\([^()@]+\)/) || parserInput.$re(/^[\.#](?=@)/) ||
                    this.entities.variableCurly();

                if (! e) {
                    parserInput.save();
                    if (parserInput.$char('(')) {
                        if ((v = this.selector()) && parserInput.$char(')')) {
                            e = new(tree.Paren)(v);
                            parserInput.forget();
                        } else {
                            parserInput.restore("Missing closing ')'");
                        }
                    } else {
                        parserInput.forget();
                    }
                }

                if (e) { return new(tree.Element)(c, e, index, env.currentFileInfo); }
            },

            //
            // Combinators combine elements together, in a Selector.
            //
            // Because our parser isn't white-space sensitive, special care
            // has to be taken, when parsing the descendant combinator, ` `,
            // as it's an empty space. We have to check the previous character
            // in the input, to see if it's a ` ` character. More info on how
            // we deal with this in *combinator.js*.
            //
            combinator: function () {
                var c = parserInput.currentChar();

                if (c === '/') {
                    parserInput.save();
                    var slashedCombinator = parserInput.$re(/^\/[a-z]+\//i);
                    if (slashedCombinator) {
                        parserInput.forget();
                        return new(tree.Combinator)(slashedCombinator);
                    }
                    parserInput.restore();
                }

                if (c === '>' || c === '+' || c === '~' || c === '|' || c === '^') {
                    parserInput.i++;
                    if (c === '^' && parserInput.currentChar() === '^') {
                        c = '^^';
                        parserInput.i++;
                    }
                    while (parserInput.isWhitespace()) { parserInput.i++; }
                    return new(tree.Combinator)(c);
                } else if (parserInput.isWhitespace(-1)) {
                    return new(tree.Combinator)(" ");
                } else {
                    return new(tree.Combinator)(null);
                }
            },
            //
            // A CSS selector (see selector below)
            // with less extensions e.g. the ability to extend and guard
            //
            lessSelector: function () {
                return this.selector(true);
            },
            //
            // A CSS Selector
            //
            //     .class > div + h1
            //     li a:hover
            //
            // Selectors are made out of one or more Elements, see above.
            //
            selector: function (isLess) {
                var index = parserInput.i, elements, extendList, c, e, extend, when, condition;

                while ((isLess && (extend = this.extend())) || (isLess && (when = parserInput.$re(/^when/))) || (e = this.element())) {
                    if (when) {
                        condition = expect(this.conditions, 'expected condition');
                    } else if (condition) {
                        error("CSS guard can only be used at the end of selector");
                    } else if (extend) {
                        if (extendList) { extendList.push(extend); } else { extendList = [ extend ]; }
                    } else {
                        if (extendList) { error("Extend can only be used at the end of selector"); }
                        c = parserInput.currentChar();
                        if (elements) { elements.push(e); } else { elements = [ e ]; }
                        e = null;
                    }
                    if (c === '{' || c === '}' || c === ';' || c === ',' || c === ')') {
                        break;
                    }
                }

                if (elements) { return new(tree.Selector)(elements, extendList, condition, index, env.currentFileInfo); }
                if (extendList) { error("Extend must be used to extend a selector, it cannot be used on its own"); }
            },
            attribute: function () {
                if (! parserInput.$char('[')) { return; }

                var entities = this.entities,
                    key, val, op;

                if (!(key = entities.variableCurly())) {
                    key = expect(/^(?:[_A-Za-z0-9-\*]*\|)?(?:[_A-Za-z0-9-]|\\.)+/);
                }

                op = parserInput.$re(/^[|~*$^]?=/);
                if (op) {
                    val = entities.quoted() || parserInput.$re(/^[0-9]+%/) || parserInput.$re(/^[\w-]+/) || entities.variableCurly();
                }

                expectChar(']');

                return new(tree.Attribute)(key, op, val);
            },

            //
            // The `block` rule is used by `ruleset` and `mixin.definition`.
            // It's a wrapper around the `primary` rule, with added `{}`.
            //
            block: function () {
                var content;
                if (parserInput.$char('{') && (content = this.primary()) && parserInput.$char('}')) {
                    return content;
                }
            },

            blockRuleset: function() {
                var block = this.block();

                if (block) {
                    block = new tree.Ruleset(null, block);
                }
                return block;
            },

            detachedRuleset: function() {
                var blockRuleset = this.blockRuleset();
                if (blockRuleset) {
                    return new tree.DetachedRuleset(blockRuleset);
                }
            },

            //
            // div, .class, body > p {...}
            //
            ruleset: function () {
                var selectors, s, rules, debugInfo;

                parserInput.save();

                if (env.dumpLineNumbers) {
                    debugInfo = getDebugInfo(parserInput.i);
                }

                while (true) {
                    s = this.lessSelector();
                    if (!s) {
                        break;
                    }
                    if (selectors) { selectors.push(s); } else { selectors = [ s ]; }
                    parserInput.commentStore.length = 0;
                    if (s.condition && selectors.length > 1) {
                        error("Guards are only currently allowed on a single selector.");
                    }
                    if (! parserInput.$char(',')) { break; }
                    if (s.condition) {
                        error("Guards are only currently allowed on a single selector.");
                    }
                    parserInput.commentStore.length = 0;
                }

                if (selectors && (rules = this.block())) {
                    parserInput.forget();
                    var ruleset = new(tree.Ruleset)(selectors, rules, env.strictImports);
                    if (env.dumpLineNumbers) {
                        ruleset.debugInfo = debugInfo;
                    }
                    return ruleset;
                } else {
                    parserInput.restore();
                }
            },
            rule: function (tryAnonymous) {
                var name, value, startOfRule = parserInput.i, c = parserInput.currentChar(), important, merge, isVariable;

                if (c === '.' || c === '#' || c === '&') { return; }

                parserInput.save();

                name = this.variable() || this.ruleProperty();
                if (name) {
                    isVariable = typeof name === "string";

                    if (isVariable) {
                        value = this.detachedRuleset();
                    }

                    parserInput.commentStore.length = 0;
                    if (!value) {
                        // a name returned by this.ruleProperty() is always an array of the form:
                        // [string-1, ..., string-n, ""] or [string-1, ..., string-n, "+"]
                        // where each item is a tree.Keyword or tree.Variable
                        merge = !isVariable && name.pop().value;

                        // prefer to try to parse first if its a variable or we are compressing
                        // but always fallback on the other one
                        var tryValueFirst = !tryAnonymous && (env.compress || isVariable);

                        if (tryValueFirst) {
                            value = this.value();
                        }
                        if (!value) {
                            value = this.anonymousValue();
                            if (value) {
                                parserInput.forget();
                                // anonymous values absorb the end ';' which is reequired for them to work
                                return new (tree.Rule)(name, value, false, merge, startOfRule, env.currentFileInfo);
                            }
                        }
                        if (!tryValueFirst && !value) {
                            value = this.value();
                        }

                        important = this.important();
                    }

                    if (value && this.end()) {
                        parserInput.forget();
                        return new (tree.Rule)(name, value, important, merge, startOfRule, env.currentFileInfo);
                    } else {
                        parserInput.restore();
                        if (value && !tryAnonymous) {
                            return this.rule(true);
                        }
                    }
                } else {
                    parserInput.forget();
                }
            },
            anonymousValue: function () {
                var match = parserInput.$re(/^([^@+\/'"*`(;{}-]*);/);
                if (match) {
                    return new(tree.Anonymous)(match[1]);
                }
            },

            //
            // An @import directive
            //
            //     @import "lib";
            //
            // Depending on our environment, importing is done differently:
            // In the browser, it's an XHR request, in Node, it would be a
            // file-system operation. The function used for importing is
            // stored in `import`, which we pass to the Import constructor.
            //
            "import": function () {
                var path, features, index = parserInput.i;

                var dir = parserInput.$re(/^@import?\s+/);

                if (dir) {
                    var options = (dir ? this.importOptions() : null) || {};

                    if ((path = this.entities.quoted() || this.entities.url())) {
                        features = this.mediaFeatures();

                        if (!parserInput.$(';')) {
                            parserInput.i = index;
                            error("missing semi-colon or unrecognised media features on import");
                        }
                        features = features && new(tree.Value)(features);
                        return new(tree.Import)(path, features, options, index, env.currentFileInfo);
                    }
                    else
                    {
                        parserInput.i = index;
                        error("malformed import statement");
                    }
                }
            },

            importOptions: function() {
                var o, options = {}, optionName, value;

                // list of options, surrounded by parens
                if (! parserInput.$char('(')) { return null; }
                do {
                    o = this.importOption();
                    if (o) {
                        optionName = o;
                        value = true;
                        switch(optionName) {
                            case "css":
                                optionName = "less";
                                value = false;
                            break;
                            case "once":
                                optionName = "multiple";
                                value = false;
                            break;
                        }
                        options[optionName] = value;
                        if (! parserInput.$char(',')) { break; }
                    }
                } while (o);
                expectChar(')');
                return options;
            },

            importOption: function() {
                var opt = parserInput.$re(/^(less|css|multiple|once|inline|reference)/);
                if (opt) {
                    return opt[1];
                }
            },

            mediaFeature: function () {
                var entities = this.entities, nodes = [], e, p;
                parserInput.save();
                do {
                    e = entities.keyword() || entities.variable();
                    if (e) {
                        nodes.push(e);
                    } else if (parserInput.$char('(')) {
                        p = this.property();
                        e = this.value();
                        if (parserInput.$char(')')) {
                            if (p && e) {
                                nodes.push(new(tree.Paren)(new(tree.Rule)(p, e, null, null, parserInput.i, env.currentFileInfo, true)));
                            } else if (e) {
                                nodes.push(new(tree.Paren)(e));
                            } else {
                                parserInput.restore("badly formed media feature definition");
                                return null;
                            }
                        } else {
                            parserInput.restore("Missing closing ')'");
                            return null;
                        }
                    }
                } while (e);

                parserInput.forget();
                if (nodes.length > 0) {
                    return new(tree.Expression)(nodes);
                }
            },

            mediaFeatures: function () {
                var entities = this.entities, features = [], e;
                do {
                    e = this.mediaFeature();
                    if (e) {
                        features.push(e);
                        if (! parserInput.$char(',')) { break; }
                    } else {
                        e = entities.variable();
                        if (e) {
                            features.push(e);
                            if (! parserInput.$char(',')) { break; }
                        }
                    }
                } while (e);

                return features.length > 0 ? features : null;
            },

            media: function () {
                var features, rules, media, debugInfo;

                if (env.dumpLineNumbers) {
                    debugInfo = getDebugInfo(parserInput.i);
                }

                if (parserInput.$re(/^@media/)) {
                    features = this.mediaFeatures();

                    rules = this.block();
                    if (rules) {
                        media = new(tree.Media)(rules, features, parserInput.i, env.currentFileInfo);
                        if (env.dumpLineNumbers) {
                            media.debugInfo = debugInfo;
                        }
                        return media;
                    }
                }
            },

            //
            // A CSS Directive
            //
            //     @charset "utf-8";
            //
            directive: function () {
                var index = parserInput.i, name, value, rules, nonVendorSpecificName,
                    hasIdentifier, hasExpression, hasUnknown, hasBlock = true;

                if (parserInput.currentChar() !== '@') { return; }

                value = this['import']() || this.media();
                if (value) {
                    return value;
                }

                parserInput.save();

                name = parserInput.$re(/^@[a-z-]+/);

                if (!name) { return; }

                nonVendorSpecificName = name;
                if (name.charAt(1) == '-' && name.indexOf('-', 2) > 0) {
                    nonVendorSpecificName = "@" + name.slice(name.indexOf('-', 2) + 1);
                }

                switch(nonVendorSpecificName) {
                    /*
                    case "@font-face":
                    case "@viewport":
                    case "@top-left":
                    case "@top-left-corner":
                    case "@top-center":
                    case "@top-right":
                    case "@top-right-corner":
                    case "@bottom-left":
                    case "@bottom-left-corner":
                    case "@bottom-center":
                    case "@bottom-right":
                    case "@bottom-right-corner":
                    case "@left-top":
                    case "@left-middle":
                    case "@left-bottom":
                    case "@right-top":
                    case "@right-middle":
                    case "@right-bottom":
                        hasBlock = true;
                        break;
                    */
                    case "@charset":
                        hasIdentifier = true;
                        hasBlock = false;
                        break;
                    case "@namespace":
                        hasExpression = true;
                        hasBlock = false;
                        break;
                    case "@keyframes":
                        hasIdentifier = true;
                        break;
                    case "@host":
                    case "@page":
                    case "@document":
                    case "@supports":
                        hasUnknown = true;
                        break;
                }

                parserInput.commentStore.length = 0;

                if (hasIdentifier) {
                    value = this.entity();
                    if (!value) {
                        error("expected " + name + " identifier");
                    }
                } else if (hasExpression) {
                    value = this.expression();
                    if (!value) {
                        error("expected " + name + " expression");
                    }
                } else if (hasUnknown) {
                    value = (parserInput.$re(/^[^{;]+/) || '').trim();
                    if (value) {
                        value = new(tree.Anonymous)(value);
                    }
                }

                if (hasBlock) {
                    rules = this.blockRuleset();
                }

                if (rules || (!hasBlock && value && parserInput.$char(';'))) {
                    parserInput.forget();
                    return new(tree.Directive)(name, value, rules, index, env.currentFileInfo,
                        env.dumpLineNumbers ? getDebugInfo(index) : null);
                }

                parserInput.restore("directive options not recognised");
            },

            //
            // A Value is a comma-delimited list of Expressions
            //
            //     font-family: Baskerville, Georgia, serif;
            //
            // In a Rule, a Value represents everything after the `:`,
            // and before the `;`.
            //
            value: function () {
                var e, expressions = [];

                do {
                    e = this.expression();
                    if (e) {
                        expressions.push(e);
                        if (! parserInput.$char(',')) { break; }
                    }
                } while(e);

                if (expressions.length > 0) {
                    return new(tree.Value)(expressions);
                }
            },
            important: function () {
                if (parserInput.currentChar() === '!') {
                    return parserInput.$re(/^! *important/);
                }
            },
            sub: function () {
                var a, e;

                if (parserInput.$char('(')) {
                    a = this.addition();
                    if (a) {
                        e = new(tree.Expression)([a]);
                        expectChar(')');
                        e.parens = true;
                        return e;
                    }
                }
            },
            multiplication: function () {
                var m, a, op, operation, isSpaced;
                m = this.operand();
                if (m) {
                    isSpaced = parserInput.isWhitespace(-1);
                    while (true) {
                        if (parserInput.peek(/^\/[*\/]/)) {
                            break;
                        }

                        parserInput.save();

                        op = parserInput.$char('/') || parserInput.$char('*');

                        if (!op) { parserInput.forget(); break; }

                        a = this.operand();

                        if (!a) { parserInput.restore(); break; }
                        parserInput.forget();

                        m.parensInOp = true;
                        a.parensInOp = true;
                        operation = new(tree.Operation)(op, [operation || m, a], isSpaced);
                        isSpaced = parserInput.isWhitespace(-1);
                    }
                    return operation || m;
                }
            },
            addition: function () {
                var m, a, op, operation, isSpaced;
                m = this.multiplication();
                if (m) {
                    isSpaced = parserInput.isWhitespace(-1);
                    while (true) {
                        op = parserInput.$re(/^[-+]\s+/) || (!isSpaced && (parserInput.$char('+') || parserInput.$char('-')));
                        if (!op) {
                            break;
                        }
                        a = this.multiplication();
                        if (!a) {
                            break;
                        }

                        m.parensInOp = true;
                        a.parensInOp = true;
                        operation = new(tree.Operation)(op, [operation || m, a], isSpaced);
                        isSpaced = parserInput.isWhitespace(-1);
                    }
                    return operation || m;
                }
            },
            conditions: function () {
                var a, b, index = parserInput.i, condition;

                a = this.condition();
                if (a) {
                    while (true) {
                        if (!parserInput.peek(/^,\s*(not\s*)?\(/) || !parserInput.$char(',')) {
                            break;
                        }
                        b = this.condition();
                        if (!b) {
                            break;
                        }
                        condition = new(tree.Condition)('or', condition || a, b, index);
                    }
                    return condition || a;
                }
            },
            condition: function () {
                var entities = this.entities, index = parserInput.i, negate = false,
                    a, b, c, op;

                if (parserInput.$re(/^not/)) { negate = true; }
                expectChar('(');
                a = this.addition() || entities.keyword() || entities.quoted();
                if (a) {
                    op = parserInput.$re(/^(?:>=|<=|=<|[<=>])/);
                    if (op) {
                        b = this.addition() || entities.keyword() || entities.quoted();
                        if (b) {
                            c = new(tree.Condition)(op, a, b, index, negate);
                        } else {
                            error('expected expression');
                        }
                    } else {
                        c = new(tree.Condition)('=', a, new(tree.Keyword)('true'), index, negate);
                    }
                    expectChar(')');
                    return parserInput.$re(/^and/) ? new(tree.Condition)('and', c, this.condition()) : c;
                }
            },

            //
            // An operand is anything that can be part of an operation,
            // such as a Color, or a Variable
            //
            operand: function () {
                var entities = this.entities, negate;

                if (parserInput.peek(/^-[@\(]/)) {
                    negate = parserInput.$char('-');
                }

                var o = this.sub() || entities.dimension() ||
                        entities.color() || entities.variable() ||
                        entities.call();

                if (negate) {
                    o.parensInOp = true;
                    o = new(tree.Negative)(o);
                }

                return o;
            },

            //
            // Expressions either represent mathematical operations,
            // or white-space delimited Entities.
            //
            //     1px solid black
            //     @var * 2
            //
            expression: function () {
                var entities = [], e, delim;

                do {
                    e = this.comment();
                    if (e) {
                        entities.push(e);
                        continue;
                    }
                    e = this.addition() || this.entity();
                    if (e) {
                        entities.push(e);
                        // operations do not allow keyword "/" dimension (e.g. small/20px) so we support that here
                        if (!parserInput.peek(/^\/[\/*]/)) {
                            delim = parserInput.$char('/');
                            if (delim) {
                                entities.push(new(tree.Anonymous)(delim));
                            }
                        }
                    }
                } while (e);
                if (entities.length > 0) {
                    return new(tree.Expression)(entities);
                }
            },
            property: function () {
                var name = parserInput.$re(/^(\*?-?[_a-zA-Z0-9-]+)\s*:/);
                if (name) {
                    return name[1];
                }
            },
            ruleProperty: function () {
                var name = [], index = [], s, k;

                parserInput.save();

                function match(re) {
                    var i = parserInput.i,
                        chunk = parserInput.$re(re);
                    if (chunk) {
                        index.push(i);
                        return name.push(chunk[1]);
                    }
                }

                match(/^(\*?)/);
                while (true) {
                    if (!match(/^((?:[\w-]+)|(?:@\{[\w-]+\}))/)) {
                        break;
                    }
                }

                if ((name.length > 1) && match(/^((?:\+_|\+)?)\s*:/)) {
                    parserInput.forget();

                    // at last, we have the complete match now. move forward,
                    // convert name particles to tree objects and return:
                    if (name[0] === '') {
                        name.shift();
                        index.shift();
                    }
                    for (k = 0; k < name.length; k++) {
                        s = name[k];
                        name[k] = (s.charAt(0) !== '@') ?
                            new(tree.Keyword)(s) :
                            new(tree.Variable)('@' + s.slice(2, -1),
                                index[k], env.currentFileInfo);
                    }
                    return name;
                }
                parserInput.restore();
            }
        }
    };

    parser.getInput = getInput;
    parser.getLocation = parserInput.getLocation;

    return parser;
};
Parser.serializeVars = function(vars) {
    var s = '';

    for (var name in vars) {
        if (Object.hasOwnProperty.call(vars, name)) {
            var value = vars[name];
            s += ((name[0] === '@') ? '' : '@') + name +': '+ value +
                ((('' + value).slice(-1) === ';') ? '' : ';');
        }
    }

    return s;
};

return Parser;
};

},{"../contexts.js":2,"../less-error.js":19,"../source-map-output":25,"../tree/index.js":43,"../visitor/index.js":66,"./imports.js":22,"./parser-input.js":23}],25:[function(require,module,exports){
module.exports = function (environment) {

    var SourceMapOutput = function (options) {
        this._css = [];
        this._rootNode = options.rootNode;
        this._writeSourceMap = options.writeSourceMap;
        this._contentsMap = options.contentsMap;
        this._contentsIgnoredCharsMap = options.contentsIgnoredCharsMap;
        this._sourceMapFilename = options.sourceMapFilename;
        this._outputFilename = options.outputFilename;
        this._sourceMapURL = options.sourceMapURL;
        if (options.sourceMapBasepath) {
            this._sourceMapBasepath = options.sourceMapBasepath.replace(/\\/g, '/');
        }
        this._sourceMapRootpath = options.sourceMapRootpath;
        this._outputSourceFiles = options.outputSourceFiles;
        this._sourceMapGeneratorConstructor = environment.getSourceMapGenerator();

        if (this._sourceMapRootpath && this._sourceMapRootpath.charAt(this._sourceMapRootpath.length-1) !== '/') {
            this._sourceMapRootpath += '/';
        }

        this._lineNumber = 0;
        this._column = 0;
    };

    SourceMapOutput.prototype.normalizeFilename = function(filename) {
        filename = filename.replace(/\\/g, '/');

        if (this._sourceMapBasepath && filename.indexOf(this._sourceMapBasepath) === 0) {
            filename = filename.substring(this._sourceMapBasepath.length);
            if (filename.charAt(0) === '\\' || filename.charAt(0) === '/') {
               filename = filename.substring(1);
            }
        }
        return (this._sourceMapRootpath || "") + filename;
    };

    SourceMapOutput.prototype.add = function(chunk, fileInfo, index, mapLines) {

        //ignore adding empty strings
        if (!chunk) {
            return;
        }

        var lines,
            sourceLines,
            columns,
            sourceColumns,
            i;

        if (fileInfo) {
            var inputSource = this._contentsMap[fileInfo.filename];

            // remove vars/banner added to the top of the file
            if (this._contentsIgnoredCharsMap[fileInfo.filename]) {
                // adjust the index
                index -= this._contentsIgnoredCharsMap[fileInfo.filename];
                if (index < 0) { index = 0; }
                // adjust the source
                inputSource = inputSource.slice(this._contentsIgnoredCharsMap[fileInfo.filename]);
            }
            inputSource = inputSource.substring(0, index);
            sourceLines = inputSource.split("\n");
            sourceColumns = sourceLines[sourceLines.length-1];
        }

        lines = chunk.split("\n");
        columns = lines[lines.length-1];

        if (fileInfo) {
            if (!mapLines) {
                this._sourceMapGenerator.addMapping({ generated: { line: this._lineNumber + 1, column: this._column},
                    original: { line: sourceLines.length, column: sourceColumns.length},
                    source: this.normalizeFilename(fileInfo.filename)});
            } else {
                for(i = 0; i < lines.length; i++) {
                    this._sourceMapGenerator.addMapping({ generated: { line: this._lineNumber + i + 1, column: i === 0 ? this._column : 0},
                        original: { line: sourceLines.length + i, column: i === 0 ? sourceColumns.length : 0},
                        source: this.normalizeFilename(fileInfo.filename)});
                }
            }
        }

        if (lines.length === 1) {
            this._column += columns.length;
        } else {
            this._lineNumber += lines.length - 1;
            this._column = columns.length;
        }

        this._css.push(chunk);
    };

    SourceMapOutput.prototype.isEmpty = function() {
        return this._css.length === 0;
    };

    SourceMapOutput.prototype.toCSS = function(env) {
        this._sourceMapGenerator = new this._sourceMapGeneratorConstructor({ file: this._outputFilename, sourceRoot: null });

        if (this._outputSourceFiles) {
            for(var filename in this._contentsMap) {
                if (this._contentsMap.hasOwnProperty(filename))
                {
                    var source = this._contentsMap[filename];
                    if (this._contentsIgnoredCharsMap[filename]) {
                        source = source.slice(this._contentsIgnoredCharsMap[filename]);
                    }
                    this._sourceMapGenerator.setSourceContent(this.normalizeFilename(filename), source);
                }
            }
        }

        this._rootNode.genCSS(env, this);

        if (this._css.length > 0) {
            var sourceMapURL,
                sourceMapContent = JSON.stringify(this._sourceMapGenerator.toJSON());

            if (this._sourceMapURL) {
                sourceMapURL = this._sourceMapURL;
            } else if (this._sourceMapFilename) {
                sourceMapURL = this.normalizeFilename(this._sourceMapFilename);
            }

            if (this._writeSourceMap) {
                this._writeSourceMap(sourceMapContent);
            } else {
                sourceMapURL = "data:application/json;base64," + environment.encodeBase64(sourceMapContent);
            }

            if (sourceMapURL) {
                this._css.push("/*# sourceMappingURL=" + sourceMapURL + " */");
            }
        }

        return this._css.join('');
    };

    return SourceMapOutput;
};

},{}],26:[function(require,module,exports){
var Node = require("./node.js");

var Alpha = function (val) {
    this.value = val;
};
Alpha.prototype = new Node();
Alpha.prototype.type = "Alpha";

Alpha.prototype.accept = function (visitor) {
    this.value = visitor.visit(this.value);
};
Alpha.prototype.eval = function (env) {
    if (this.value.eval) { return new Alpha(this.value.eval(env)); }
    return this;
};
Alpha.prototype.genCSS = function (env, output) {
    output.add("alpha(opacity=");

    if (this.value.genCSS) {
        this.value.genCSS(env, output);
    } else {
        output.add(this.value);
    }

    output.add(")");
};

module.exports = Alpha;

},{"./node.js":51}],27:[function(require,module,exports){
var Node = require("./node.js");

var Anonymous = function (value, index, currentFileInfo, mapLines, rulesetLike) {
    this.value = value;
    this.index = index;
    this.mapLines = mapLines;
    this.currentFileInfo = currentFileInfo;
    this.rulesetLike = (typeof rulesetLike === 'undefined')? false : rulesetLike;
};
Anonymous.prototype = new Node();
Anonymous.prototype.type = "Anonymous";
Anonymous.prototype.eval = function () {
    return new Anonymous(this.value, this.index, this.currentFileInfo, this.mapLines, this.rulesetLike);
};
Anonymous.prototype.compare = function (x) {
    if (!x.toCSS) {
        return -1;
    }

    var left = this.toCSS(),
        right = x.toCSS();

    if (left === right) {
        return 0;
    }

    return left < right ? -1 : 1;
};
Anonymous.prototype.isRulesetLike = function() {
    return this.rulesetLike;
};
Anonymous.prototype.genCSS = function (env, output) {
    output.add(this.value, this.currentFileInfo, this.index, this.mapLines);
};
module.exports = Anonymous;

},{"./node.js":51}],28:[function(require,module,exports){
var Node = require("./node.js");

var Assignment = function (key, val) {
    this.key = key;
    this.value = val;
};

Assignment.prototype = new Node();
Assignment.prototype.type = "Assignment";
Assignment.prototype.accept = function (visitor) {
    this.value = visitor.visit(this.value);
};
Assignment.prototype.eval = function (env) {
    if (this.value.eval) {
        return new(Assignment)(this.key, this.value.eval(env));
    }
    return this;
};
Assignment.prototype.genCSS = function (env, output) {
    output.add(this.key + '=');
    if (this.value.genCSS) {
        this.value.genCSS(env, output);
    } else {
        output.add(this.value);
    }
};
module.exports = Assignment;


},{"./node.js":51}],29:[function(require,module,exports){
var Node = require("./node.js");

var Attribute = function (key, op, value) {
    this.key = key;
    this.op = op;
    this.value = value;
};
Attribute.prototype = new Node();
Attribute.prototype.type = "Attribute";
Attribute.prototype.eval = function (env) {
    return new(Attribute)(this.key.eval ? this.key.eval(env) : this.key,
        this.op, (this.value && this.value.eval) ? this.value.eval(env) : this.value);
};
Attribute.prototype.genCSS = function (env, output) {
    output.add(this.toCSS(env));
};
Attribute.prototype.toCSS = function (env) {
    var value = this.key.toCSS ? this.key.toCSS(env) : this.key;

    if (this.op) {
        value += this.op;
        value += (this.value.toCSS ? this.value.toCSS(env) : this.value);
    }

    return '[' + value + ']';
};
module.exports = Attribute;

},{"./node.js":51}],30:[function(require,module,exports){
var Node = require("./node.js"),
    FunctionCaller = require("../functions/function-caller.js");
//
// A function call node.
//
var Call = function (name, args, index, currentFileInfo) {
    this.name = name;
    this.args = args;
    this.index = index;
    this.currentFileInfo = currentFileInfo;
};
Call.prototype = new Node();
Call.prototype.type = "Call";
Call.prototype.accept = function (visitor) {
    if (this.args) {
        this.args = visitor.visitArray(this.args);
    }
};
//
// When evaluating a function call,
// we either find the function in `less.functions` [1],
// in which case we call it, passing the  evaluated arguments,
// if this returns null or we cannot find the function, we
// simply print it out as it appeared originally [2].
//
// The *functions.js* file contains the built-in functions.
//
// The reason why we evaluate the arguments, is in the case where
// we try to pass a variable to a function, like: `saturate(@color)`.
// The function should receive the value, not the variable.
//
Call.prototype.eval = function (env) {
    var args = this.args.map(function (a) { return a.eval(env); }),
        result, funcCaller = new FunctionCaller(this.name, env, this.currentFileInfo);

    if (funcCaller.isValid()) { // 1.
        try {
            result = funcCaller.call(args);
            if (result != null) {
                return result;
            }
        } catch (e) {
            throw { type: e.type || "Runtime",
                    message: "error evaluating function `" + this.name + "`" +
                             (e.message ? ': ' + e.message : ''),
                    index: this.index, filename: this.currentFileInfo.filename };
        }
    }

    return new Call(this.name, args, this.index, this.currentFileInfo);
};
Call.prototype.genCSS = function (env, output) {
    output.add(this.name + "(", this.currentFileInfo, this.index);

    for(var i = 0; i < this.args.length; i++) {
        this.args[i].genCSS(env, output);
        if (i + 1 < this.args.length) {
            output.add(", ");
        }
    }

    output.add(")");
};
module.exports = Call;

},{"../functions/function-caller.js":11,"./node.js":51}],31:[function(require,module,exports){
var Node = require("./node.js"),
    colors = require("../data/colors.js");

//
// RGB Colors - #ff0014, #eee
//
var Color = function (rgb, a) {
    //
    // The end goal here, is to parse the arguments
    // into an integer triplet, such as `128, 255, 0`
    //
    // This facilitates operations and conversions.
    //
    if (Array.isArray(rgb)) {
        this.rgb = rgb;
    } else if (rgb.length == 6) {
        this.rgb = rgb.match(/.{2}/g).map(function (c) {
            return parseInt(c, 16);
        });
    } else {
        this.rgb = rgb.split('').map(function (c) {
            return parseInt(c + c, 16);
        });
    }
    this.alpha = typeof(a) === 'number' ? a : 1;
};

Color.prototype = new Node();
Color.prototype.type = "Color";

function clamp(v, max) {
    return Math.min(Math.max(v, 0), max);
}

function toHex(v) {
    return '#' + v.map(function (c) {
        c = clamp(Math.round(c), 255);
        return (c < 16 ? '0' : '') + c.toString(16);
    }).join('');
}

Color.prototype.luma = function () {
    var r = this.rgb[0] / 255,
        g = this.rgb[1] / 255,
        b = this.rgb[2] / 255;

    r = (r <= 0.03928) ? r / 12.92 : Math.pow(((r + 0.055) / 1.055), 2.4);
    g = (g <= 0.03928) ? g / 12.92 : Math.pow(((g + 0.055) / 1.055), 2.4);
    b = (b <= 0.03928) ? b / 12.92 : Math.pow(((b + 0.055) / 1.055), 2.4);

    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
Color.prototype.genCSS = function (env, output) {
    output.add(this.toCSS(env));
};
Color.prototype.toCSS = function (env, doNotCompress) {
    var compress = env && env.compress && !doNotCompress, color, alpha;

    // `keyword` is set if this color was originally
    // converted from a named color string so we need
    // to respect this and try to output named color too.
    if (this.keyword) {
        return this.keyword;
    }

    // If we have some transparency, the only way to represent it
    // is via `rgba`. Otherwise, we use the hex representation,
    // which has better compatibility with older browsers.
    // Values are capped between `0` and `255`, rounded and zero-padded.
    alpha = this.fround(env, this.alpha);
    if (alpha < 1) {
        return "rgba(" + this.rgb.map(function (c) {
            return clamp(Math.round(c), 255);
        }).concat(clamp(alpha, 1))
            .join(',' + (compress ? '' : ' ')) + ")";
    }

    color = this.toRGB();

    if (compress) {
        var splitcolor = color.split('');

        // Convert color to short format
        if (splitcolor[1] === splitcolor[2] && splitcolor[3] === splitcolor[4] && splitcolor[5] === splitcolor[6]) {
            color = '#' + splitcolor[1] + splitcolor[3] + splitcolor[5];
        }
    }

    return color;
};

//
// Operations have to be done per-channel, if not,
// channels will spill onto each other. Once we have
// our result, in the form of an integer triplet,
// we create a new Color node to hold the result.
//
Color.prototype.operate = function (env, op, other) {
    var rgb = [];
    var alpha = this.alpha * (1 - other.alpha) + other.alpha;
    for (var c = 0; c < 3; c++) {
        rgb[c] = this._operate(env, op, this.rgb[c], other.rgb[c]);
    }
    return new(Color)(rgb, alpha);
};
Color.prototype.toRGB = function () {
    return toHex(this.rgb);
};
Color.prototype.toHSL = function () {
    var r = this.rgb[0] / 255,
        g = this.rgb[1] / 255,
        b = this.rgb[2] / 255,
        a = this.alpha;

    var max = Math.max(r, g, b), min = Math.min(r, g, b);
    var h, s, l = (max + min) / 2, d = max - min;

    if (max === min) {
        h = s = 0;
    } else {
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2;               break;
            case b: h = (r - g) / d + 4;               break;
        }
        h /= 6;
    }
    return { h: h * 360, s: s, l: l, a: a };
};
//Adapted from http://mjijackson.com/2008/02/rgb-to-hsl-and-rgb-to-hsv-color-model-conversion-algorithms-in-javascript
Color.prototype.toHSV = function () {
    var r = this.rgb[0] / 255,
        g = this.rgb[1] / 255,
        b = this.rgb[2] / 255,
        a = this.alpha;

    var max = Math.max(r, g, b), min = Math.min(r, g, b);
    var h, s, v = max;

    var d = max - min;
    if (max === 0) {
        s = 0;
    } else {
        s = d / max;
    }

    if (max === min) {
        h = 0;
    } else {
        switch(max){
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    return { h: h * 360, s: s, v: v, a: a };
};
Color.prototype.toARGB = function () {
    return toHex([this.alpha * 255].concat(this.rgb));
};
Color.prototype.compare = function (x) {
    if (!x.rgb) {
        return -1;
    }

    return (x.rgb[0] === this.rgb[0] &&
        x.rgb[1] === this.rgb[1] &&
        x.rgb[2] === this.rgb[2] &&
        x.alpha === this.alpha) ? 0 : -1;
};

Color.fromKeyword = function(keyword) {
    var c, key = keyword.toLowerCase();
    if (colors.hasOwnProperty(key)) {
        c = new(Color)(colors[key].slice(1));
    }
    else if (key === "transparent") {
        c = new(Color)([0, 0, 0], 0);
    }

    if (c) {
        c.keyword = keyword;
        return c;
    }
};
module.exports = Color;

},{"../data/colors.js":3,"./node.js":51}],32:[function(require,module,exports){
var Node = require("./node.js");

var Combinator = function (value) {
    if (value === ' ') {
        this.value = ' ';
    } else {
        this.value = value ? value.trim() : "";
    }
};
Combinator.prototype = new Node();
Combinator.prototype.type = "Combinator";
var _noSpaceCombinators = {
    '': true,
    ' ': true,
    '|': true
};
Combinator.prototype.genCSS = function (env, output) {
    var spaceOrEmpty = (env.compress || _noSpaceCombinators[this.value]) ? '' : ' ';
    output.add(spaceOrEmpty + this.value + spaceOrEmpty);
};
module.exports = Combinator;

},{"./node.js":51}],33:[function(require,module,exports){
var Node = require("./node.js"),
    getDebugInfo = require("./debug-info.js");

var Comment = function (value, isLineComment, index, currentFileInfo) {
    this.value = value;
    this.isLineComment = isLineComment;
    this.currentFileInfo = currentFileInfo;
};
Comment.prototype = new Node();
Comment.prototype.type = "Comment";
Comment.prototype.genCSS = function (env, output) {
    if (this.debugInfo) {
        output.add(getDebugInfo(env, this), this.currentFileInfo, this.index);
    }
    output.add(this.value);
};
Comment.prototype.isSilent = function(env) {
    var isReference = (this.currentFileInfo && this.currentFileInfo.reference && !this.isReferenced),
        isCompressed = env.compress && this.value[2] !== "!";
    return this.isLineComment || isReference || isCompressed;
};
Comment.prototype.markReferenced = function () {
    this.isReferenced = true;
};
Comment.prototype.isRulesetLike = function(root) {
    return Boolean(root);
};
module.exports = Comment;

},{"./debug-info.js":35,"./node.js":51}],34:[function(require,module,exports){
var Node = require("./node.js");

var Condition = function (op, l, r, i, negate) {
    this.op = op.trim();
    this.lvalue = l;
    this.rvalue = r;
    this.index = i;
    this.negate = negate;
};
Condition.prototype = new Node();
Condition.prototype.type = "Condition";
Condition.prototype.accept = function (visitor) {
    this.lvalue = visitor.visit(this.lvalue);
    this.rvalue = visitor.visit(this.rvalue);
};
Condition.prototype.eval = function (env) {
    var a = this.lvalue.eval(env),
        b = this.rvalue.eval(env);

    var i = this.index, result;

    result = (function (op) {
        switch (op) {
            case 'and':
                return a && b;
            case 'or':
                return a || b;
            default:
                if (a.compare) {
                    result = a.compare(b);
                } else if (b.compare) {
                    result = b.compare(a);
                } else {
                    throw { type: "Type",
                            message: "Unable to perform comparison",
                            index: i };
                }
                switch (result) {
                    case -1: return op === '<' || op === '=<' || op === '<=';
                    case  0: return op === '=' || op === '>=' || op === '=<' || op === '<=';
                    case  1: return op === '>' || op === '>=';
                }
        }
    })(this.op);
    return this.negate ? !result : result;
};
module.exports = Condition;

},{"./node.js":51}],35:[function(require,module,exports){
var debugInfo = function(env, ctx, lineSeperator) {
    var result="";
    if (env.dumpLineNumbers && !env.compress) {
        switch(env.dumpLineNumbers) {
            case 'comments':
                result = debugInfo.asComment(ctx);
                break;
            case 'mediaquery':
                result = debugInfo.asMediaQuery(ctx);
                break;
            case 'all':
                result = debugInfo.asComment(ctx) + (lineSeperator || "") + debugInfo.asMediaQuery(ctx);
                break;
        }
    }
    return result;
};

debugInfo.asComment = function(ctx) {
    return '/* line ' + ctx.debugInfo.lineNumber + ', ' + ctx.debugInfo.fileName + ' */\n';
};

debugInfo.asMediaQuery = function(ctx) {
    return '@media -sass-debug-info{filename{font-family:' +
        ('file://' + ctx.debugInfo.fileName).replace(/([.:\/\\])/g, function (a) {
            if (a == '\\') {
                a = '\/';
            }
            return '\\' + a;
        }) +
        '}line{font-family:\\00003' + ctx.debugInfo.lineNumber + '}}\n';
};

module.exports = debugInfo;

},{}],36:[function(require,module,exports){
var Node = require("./node.js"),
    contexts = require("../contexts.js");

var DetachedRuleset = function (ruleset, frames) {
    this.ruleset = ruleset;
    this.frames = frames;
};
DetachedRuleset.prototype = new Node();
DetachedRuleset.prototype.type = "DetachedRuleset";
DetachedRuleset.prototype.evalFirst = true;
DetachedRuleset.prototype.accept = function (visitor) {
    this.ruleset = visitor.visit(this.ruleset);
};
DetachedRuleset.prototype.eval = function (env) {
    var frames = this.frames || env.frames.slice(0);
    return new DetachedRuleset(this.ruleset, frames);
};
DetachedRuleset.prototype.callEval = function (env) {
    return this.ruleset.eval(this.frames ? new(contexts.evalEnv)(env, this.frames.concat(env.frames)) : env);
};
module.exports = DetachedRuleset;

},{"../contexts.js":2,"./node.js":51}],37:[function(require,module,exports){
var Node = require("./node.js"),
    unitConversions = require("../data/unit-conversions.js"),
    Unit = require("./unit.js"),
    Color = require("./color.js");

//
// A number with a unit
//
var Dimension = function (value, unit) {
    this.value = parseFloat(value);
    this.unit = (unit && unit instanceof Unit) ? unit :
      new(Unit)(unit ? [unit] : undefined);
};

Dimension.prototype = new Node();
Dimension.prototype.type = "Dimension";
Dimension.prototype.accept = function (visitor) {
    this.unit = visitor.visit(this.unit);
};
Dimension.prototype.eval = function (env) {
    return this;
};
Dimension.prototype.toColor = function () {
    return new(Color)([this.value, this.value, this.value]);
};
Dimension.prototype.genCSS = function (env, output) {
    if ((env && env.strictUnits) && !this.unit.isSingular()) {
        throw new Error("Multiple units in dimension. Correct the units or use the unit function. Bad unit: "+this.unit.toString());
    }

    var value = this.fround(env, this.value),
        strValue = String(value);

    if (value !== 0 && value < 0.000001 && value > -0.000001) {
        // would be output 1e-6 etc.
        strValue = value.toFixed(20).replace(/0+$/, "");
    }

    if (env && env.compress) {
        // Zero values doesn't need a unit
        if (value === 0 && this.unit.isLength()) {
            output.add(strValue);
            return;
        }

        // Float values doesn't need a leading zero
        if (value > 0 && value < 1) {
            strValue = (strValue).substr(1);
        }
    }

    output.add(strValue);
    this.unit.genCSS(env, output);
};

// In an operation between two Dimensions,
// we default to the first Dimension's unit,
// so `1px + 2` will yield `3px`.
Dimension.prototype.operate = function (env, op, other) {
    /*jshint noempty:false */
    var value = this._operate(env, op, this.value, other.value),
        unit = this.unit.clone();

    if (op === '+' || op === '-') {
        if (unit.numerator.length === 0 && unit.denominator.length === 0) {
            unit.numerator = other.unit.numerator.slice(0);
            unit.denominator = other.unit.denominator.slice(0);
        } else if (other.unit.numerator.length === 0 && unit.denominator.length === 0) {
            // do nothing
        } else {
            other = other.convertTo(this.unit.usedUnits());

            if(env.strictUnits && other.unit.toString() !== unit.toString()) {
              throw new Error("Incompatible units. Change the units or use the unit function. Bad units: '" + unit.toString() +
                "' and '" + other.unit.toString() + "'.");
            }

            value = this._operate(env, op, this.value, other.value);
        }
    } else if (op === '*') {
        unit.numerator = unit.numerator.concat(other.unit.numerator).sort();
        unit.denominator = unit.denominator.concat(other.unit.denominator).sort();
        unit.cancel();
    } else if (op === '/') {
        unit.numerator = unit.numerator.concat(other.unit.denominator).sort();
        unit.denominator = unit.denominator.concat(other.unit.numerator).sort();
        unit.cancel();
    }
    return new(Dimension)(value, unit);
};
Dimension.prototype.compare = function (other) {
    if (other instanceof Dimension) {
        var a, b,
            aValue, bValue;

        if (this.unit.isEmpty() || other.unit.isEmpty()) {
            a = this;
            b = other;
        } else {
            a = this.unify();
            b = other.unify();
            if (a.unit.compare(b.unit) !== 0) {
                return -1;
            }
        }
        aValue = a.value;
        bValue = b.value;

        if (bValue > aValue) {
            return -1;
        } else if (bValue < aValue) {
            return 1;
        } else {
            return 0;
        }
    } else {
        return -1;
    }
};
Dimension.prototype.unify = function () {
    return this.convertTo({ length: 'px', duration: 's', angle: 'rad' });
};
Dimension.prototype.convertTo = function (conversions) {
    var value = this.value, unit = this.unit.clone(),
        i, groupName, group, targetUnit, derivedConversions = {}, applyUnit;

    if (typeof conversions === 'string') {
        for(i in unitConversions) {
            if (unitConversions[i].hasOwnProperty(conversions)) {
                derivedConversions = {};
                derivedConversions[i] = conversions;
            }
        }
        conversions = derivedConversions;
    }
    applyUnit = function (atomicUnit, denominator) {
      /*jshint loopfunc:true */
        if (group.hasOwnProperty(atomicUnit)) {
            if (denominator) {
                value = value / (group[atomicUnit] / group[targetUnit]);
            } else {
                value = value * (group[atomicUnit] / group[targetUnit]);
            }

            return targetUnit;
        }

        return atomicUnit;
    };

    for (groupName in conversions) {
        if (conversions.hasOwnProperty(groupName)) {
            targetUnit = conversions[groupName];
            group = unitConversions[groupName];

            unit.map(applyUnit);
        }
    }

    unit.cancel();

    return new(Dimension)(value, unit);
};
module.exports = Dimension;

},{"../data/unit-conversions.js":5,"./color.js":31,"./node.js":51,"./unit.js":60}],38:[function(require,module,exports){
var Node = require("./node.js"),
    Ruleset = require("./ruleset.js");

var Directive = function (name, value, rules, index, currentFileInfo, debugInfo) {
    this.name  = name;
    this.value = value;
    if (rules) {
        this.rules = rules;
        this.rules.allowImports = true;
    }
    this.index = index;
    this.currentFileInfo = currentFileInfo;
    this.debugInfo = debugInfo;
};

Directive.prototype = new Node();
Directive.prototype.type = "Directive";
Directive.prototype.accept = function (visitor) {
    var value = this.value, rules = this.rules;
    if (rules) {
        rules = visitor.visit(rules);
    }
    if (value) {
        value = visitor.visit(value);
    }
};
Directive.prototype.isRulesetLike = function() {
    return this.rules || !this.isCharset();
};
Directive.prototype.isCharset = function() {
    return "@charset" === this.name;
};
Directive.prototype.genCSS = function (env, output) {
    var value = this.value, rules = this.rules;
    output.add(this.name, this.currentFileInfo, this.index);
    if (value) {
        output.add(' ');
        value.genCSS(env, output);
    }
    if (rules) {
        this.outputRuleset(env, output, [rules]);
    } else {
        output.add(';');
    }
};
Directive.prototype.eval = function (env) {
    var value = this.value, rules = this.rules;
    if (value) {
        value = value.eval(env);
    }
    if (rules) {
        rules = rules.eval(env);
        rules.root = true;
    }
    return new(Directive)(this.name, value, rules,
        this.index, this.currentFileInfo, this.debugInfo);
};
Directive.prototype.variable = function (name) { if (this.rules) return Ruleset.prototype.variable.call(this.rules, name); };
Directive.prototype.find = function () { if (this.rules) return Ruleset.prototype.find.apply(this.rules, arguments); };
Directive.prototype.rulesets = function () { if (this.rules) return Ruleset.prototype.rulesets.apply(this.rules); };
Directive.prototype.markReferenced = function () {
    var i, rules;
    this.isReferenced = true;
    if (this.rules) {
        rules = this.rules.rules;
        for (i = 0; i < rules.length; i++) {
            if (rules[i].markReferenced) {
                rules[i].markReferenced();
            }
        }
    }
};
Directive.prototype.outputRuleset = function (env, output, rules) {
    var ruleCnt = rules.length, i;
    env.tabLevel = (env.tabLevel | 0) + 1;

    // Compressed
    if (env.compress) {
        output.add('{');
        for (i = 0; i < ruleCnt; i++) {
            rules[i].genCSS(env, output);
        }
        output.add('}');
        env.tabLevel--;
        return;
    }

    // Non-compressed
    var tabSetStr = '\n' + Array(env.tabLevel).join("  "), tabRuleStr = tabSetStr + "  ";
    if (!ruleCnt) {
        output.add(" {" + tabSetStr + '}');
    } else {
        output.add(" {" + tabRuleStr);
        rules[0].genCSS(env, output);
        for (i = 1; i < ruleCnt; i++) {
            output.add(tabRuleStr);
            rules[i].genCSS(env, output);
        }
        output.add(tabSetStr + '}');
    }

    env.tabLevel--;
};
module.exports = Directive;

},{"./node.js":51,"./ruleset.js":57}],39:[function(require,module,exports){
var Node = require("./node.js"),
    Combinator = require("./combinator.js");

var Element = function (combinator, value, index, currentFileInfo) {
    this.combinator = combinator instanceof Combinator ?
                      combinator : new(Combinator)(combinator);

    if (typeof(value) === 'string') {
        this.value = value.trim();
    } else if (value) {
        this.value = value;
    } else {
        this.value = "";
    }
    this.index = index;
    this.currentFileInfo = currentFileInfo;
};
Element.prototype = new Node();
Element.prototype.type = "Element";
Element.prototype.accept = function (visitor) {
    var value = this.value;
    this.combinator = visitor.visit(this.combinator);
    if (typeof value === "object") {
        this.value = visitor.visit(value);
    }
};
Element.prototype.eval = function (env) {
    return new(Element)(this.combinator,
                             this.value.eval ? this.value.eval(env) : this.value,
                             this.index,
                             this.currentFileInfo);
};
Element.prototype.genCSS = function (env, output) {
    output.add(this.toCSS(env), this.currentFileInfo, this.index);
};
Element.prototype.toCSS = function (env) {
    var value = (this.value.toCSS ? this.value.toCSS(env) : this.value);
    if (value === '' && this.combinator.value.charAt(0) === '&') {
        return '';
    } else {
        return this.combinator.toCSS(env || {}) + value;
    }
};
module.exports = Element;

},{"./combinator.js":32,"./node.js":51}],40:[function(require,module,exports){
var Node = require("./node.js"),
    Paren = require("./paren.js"),
    Comment = require("./comment.js");

var Expression = function (value) { this.value = value; };
Expression.prototype = new Node();
Expression.prototype.type = "Expression";
Expression.prototype.accept = function (visitor) {
    if (this.value) {
        this.value = visitor.visitArray(this.value);
    }
};
Expression.prototype.eval = function (env) {
    var returnValue,
        inParenthesis = this.parens && !this.parensInOp,
        doubleParen = false;
    if (inParenthesis) {
        env.inParenthesis();
    }
    if (this.value.length > 1) {
        returnValue = new(Expression)(this.value.map(function (e) {
            return e.eval(env);
        }));
    } else if (this.value.length === 1) {
        if (this.value[0].parens && !this.value[0].parensInOp) {
            doubleParen = true;
        }
        returnValue = this.value[0].eval(env);
    } else {
        returnValue = this;
    }
    if (inParenthesis) {
        env.outOfParenthesis();
    }
    if (this.parens && this.parensInOp && !(env.isMathOn()) && !doubleParen) {
        returnValue = new(Paren)(returnValue);
    }
    return returnValue;
};
Expression.prototype.genCSS = function (env, output) {
    for(var i = 0; i < this.value.length; i++) {
        this.value[i].genCSS(env, output);
        if (i + 1 < this.value.length) {
            output.add(" ");
        }
    }
};
Expression.prototype.throwAwayComments = function () {
    this.value = this.value.filter(function(v) {
        return !(v instanceof Comment);
    });
};
module.exports = Expression;

},{"./comment.js":33,"./node.js":51,"./paren.js":53}],41:[function(require,module,exports){
var Node = require("./node.js");

var Extend = function Extend(selector, option, index) {
    this.selector = selector;
    this.option = option;
    this.index = index;
    this.object_id = Extend.next_id++;
    this.parent_ids = [this.object_id];

    switch(option) {
        case "all":
            this.allowBefore = true;
            this.allowAfter = true;
        break;
        default:
            this.allowBefore = false;
            this.allowAfter = false;
        break;
    }
};
Extend.next_id = 0;

Extend.prototype = new Node();
Extend.prototype.type = "Extend";
Extend.prototype.accept = function (visitor) {
    this.selector = visitor.visit(this.selector);
};
Extend.prototype.eval = function (env) {
    return new(Extend)(this.selector.eval(env), this.option, this.index);
};
Extend.prototype.clone = function (env) {
    return new(Extend)(this.selector, this.option, this.index);
};
Extend.prototype.findSelfSelectors = function (selectors) {
    var selfElements = [],
        i,
        selectorElements;

    for(i = 0; i < selectors.length; i++) {
        selectorElements = selectors[i].elements;
        // duplicate the logic in genCSS function inside the selector node.
        // future TODO - move both logics into the selector joiner visitor
        if (i > 0 && selectorElements.length && selectorElements[0].combinator.value === "") {
            selectorElements[0].combinator.value = ' ';
        }
        selfElements = selfElements.concat(selectors[i].elements);
    }

    this.selfSelectors = [{ elements: selfElements }];
};
module.exports = Extend;

},{"./node.js":51}],42:[function(require,module,exports){
var Node = require("./node.js"),
    Media = require("./media.js"),
    URL = require("./url.js"),
    Quoted = require("./quoted.js"),
    Ruleset = require("./ruleset.js"),
    Anonymous = require("./anonymous.js");

//
// CSS @import node
//
// The general strategy here is that we don't want to wait
// for the parsing to be completed, before we start importing
// the file. That's because in the context of a browser,
// most of the time will be spent waiting for the server to respond.
//
// On creation, we push the import path to our import queue, though
// `import,push`, we also pass it a callback, which it'll call once
// the file has been fetched, and parsed.
//
var Import = function (path, features, options, index, currentFileInfo) {
    this.options = options;
    this.index = index;
    this.path = path;
    this.features = features;
    this.currentFileInfo = currentFileInfo;

    if (this.options.less !== undefined || this.options.inline) {
        this.css = !this.options.less || this.options.inline;
    } else {
        var pathValue = this.getPath();
        if (pathValue && /css([\?;].*)?$/.test(pathValue)) {
            this.css = true;
        }
    }
};

//
// The actual import node doesn't return anything, when converted to CSS.
// The reason is that it's used at the evaluation stage, so that the rules
// it imports can be treated like any other rules.
//
// In `eval`, we make sure all Import nodes get evaluated, recursively, so
// we end up with a flat structure, which can easily be imported in the parent
// ruleset.
//
Import.prototype = new Node();
Import.prototype.type = "Import";
Import.prototype.accept = function (visitor) {
    if (this.features) {
        this.features = visitor.visit(this.features);
    }
    this.path = visitor.visit(this.path);
    if (!this.options.inline && this.root) {
        this.root = visitor.visit(this.root);
    }
};
Import.prototype.genCSS = function (env, output) {
    if (this.css) {
        output.add("@import ", this.currentFileInfo, this.index);
        this.path.genCSS(env, output);
        if (this.features) {
            output.add(" ");
            this.features.genCSS(env, output);
        }
        output.add(';');
    }
};
Import.prototype.getPath = function () {
    if (this.path instanceof Quoted) {
        var path = this.path.value;
        return (this.css !== undefined || /(\.[a-z]*$)|([\?;].*)$/.test(path)) ? path : path + '.less';
    } else if (this.path instanceof URL) {
        return this.path.value.value;
    }
    return null;
};
Import.prototype.evalForImport = function (env) {
    return new(Import)(this.path.eval(env), this.features, this.options, this.index, this.currentFileInfo);
};
Import.prototype.evalPath = function (env) {
    var path = this.path.eval(env);
    var rootpath = this.currentFileInfo && this.currentFileInfo.rootpath;

    if (!(path instanceof URL)) {
        if (rootpath) {
            var pathValue = path.value;
            // Add the base path if the import is relative
            if (pathValue && env.isPathRelative(pathValue)) {
                path.value = rootpath +pathValue;
            }
        }
        path.value = env.normalizePath(path.value);
    }

    return path;
};
Import.prototype.eval = function (env) {
    var ruleset, features = this.features && this.features.eval(env);

    if (this.skip) {
        if (typeof this.skip === "function") {
            this.skip = this.skip();
        }
        if (this.skip) {
            return [];
        }
    }

    if (this.options.inline) {
        //todo needs to reference css file not import
        var contents = new(Anonymous)(this.root, 0, {filename: this.importedFilename}, true, true);
        return this.features ? new(Media)([contents], this.features.value) : [contents];
    } else if (this.css) {
        var newImport = new(Import)(this.evalPath(env), features, this.options, this.index);
        if (!newImport.css && this.error) {
            throw this.error;
        }
        return newImport;
    } else {
        ruleset = new(Ruleset)(null, this.root.rules.slice(0));

        ruleset.evalImports(env);

        return this.features ? new(Media)(ruleset.rules, this.features.value) : ruleset.rules;
    }
};
module.exports = Import;

},{"./anonymous.js":27,"./media.js":47,"./node.js":51,"./quoted.js":54,"./ruleset.js":57,"./url.js":61}],43:[function(require,module,exports){
var tree = {};

tree.Alpha = require('./alpha');
tree.Color = require('./color');
tree.Directive = require('./directive');
tree.DetachedRuleset = require('./detached-ruleset');
tree.Operation = require('./operation');
tree.Dimension = require('./dimension');
tree.Unit = require('./unit');
tree.Keyword = require('./keyword');
tree.Variable = require('./variable');
tree.Ruleset = require('./ruleset');
tree.Element = require('./element');
tree.Attribute = require('./attribute');
tree.Combinator = require('./combinator');
tree.Selector = require('./selector');
tree.Quoted = require('./quoted');
tree.Expression = require('./expression');
tree.Rule = require('./rule');
tree.Call = require('./call');
tree.URL = require('./url');
tree.Import = require('./import');
tree.mixin = {
    Call: require('./mixin-call'),
    Definition: require('./mixin-definition')
};
tree.Comment = require('./comment');
tree.Anonymous = require('./anonymous');
tree.Value = require('./value');
tree.JavaScript = require('./javascript');
tree.Assignment = require('./assignment');
tree.Condition = require('./condition');
tree.Paren = require('./paren');
tree.Media = require('./media');
tree.UnicodeDescriptor = require('./unicode-descriptor');
tree.Negative = require('./negative');
tree.Extend = require('./extend');
tree.RulesetCall = require('./ruleset-call');

module.exports = tree;

},{"./alpha":26,"./anonymous":27,"./assignment":28,"./attribute":29,"./call":30,"./color":31,"./combinator":32,"./comment":33,"./condition":34,"./detached-ruleset":36,"./dimension":37,"./directive":38,"./element":39,"./expression":40,"./extend":41,"./import":42,"./javascript":44,"./keyword":46,"./media":47,"./mixin-call":48,"./mixin-definition":49,"./negative":50,"./operation":52,"./paren":53,"./quoted":54,"./rule":55,"./ruleset":57,"./ruleset-call":56,"./selector":58,"./unicode-descriptor":59,"./unit":60,"./url":61,"./value":62,"./variable":63}],44:[function(require,module,exports){
var JsEvalNode = require("./js-eval-node.js"),
    Dimension = require("./dimension.js"),
    Quoted = require("./quoted.js"),
    Anonymous = require("./anonymous.js");

var JavaScript = function (string, index, escaped) {
    this.escaped = escaped;
    this.expression = string;
    this.index = index;
};
JavaScript.prototype = new JsEvalNode();
JavaScript.prototype.type = "JavaScript";
JavaScript.prototype.eval = function(env) {
    var result = this.evaluateJavaScript(this.expression, env);

    if (typeof(result) === 'number') {
        return new(Dimension)(result);
    } else if (typeof(result) === 'string') {
        return new(Quoted)('"' + result + '"', result, this.escaped, this.index);
    } else if (Array.isArray(result)) {
        return new(Anonymous)(result.join(', '));
    } else {
        return new(Anonymous)(result);
    }
};

module.exports = JavaScript;

},{"./anonymous.js":27,"./dimension.js":37,"./js-eval-node.js":45,"./quoted.js":54}],45:[function(require,module,exports){
var Node = require("./node.js"),
    Variable = require("./variable.js");

var jsEvalNode = function() {
};
jsEvalNode.prototype = new Node();

jsEvalNode.prototype.evaluateJavaScript = function (expression, env) {
    var result,
        that = this,
        context = {};

    if (env.javascriptEnabled !== undefined && !env.javascriptEnabled) {
        throw { message: "You are using JavaScript, which has been disabled." ,
            index: this.index };
    }

    expression = expression.replace(/@\{([\w-]+)\}/g, function (_, name) {
        return that.jsify(new(Variable)('@' + name, that.index).eval(env));
    });

    try {
        expression = new(Function)('return (' + expression + ')');
    } catch (e) {
        throw { message: "JavaScript evaluation error: " + e.message + " from `" + expression + "`" ,
            index: this.index };
    }

    var variables = env.frames[0].variables();
    for (var k in variables) {
        if (variables.hasOwnProperty(k)) {
            /*jshint loopfunc:true */
            context[k.slice(1)] = {
                value: variables[k].value,
                toJS: function () {
                    return this.value.eval(env).toCSS();
                }
            };
        }
    }

    try {
        result = expression.call(context);
    } catch (e) {
        throw { message: "JavaScript evaluation error: '" + e.name + ': ' + e.message.replace(/["]/g, "'") + "'" ,
            index: this.index };
    }
    return result;
};
jsEvalNode.prototype.jsify = function (obj) {
    if (Array.isArray(obj.value) && (obj.value.length > 1)) {
        return '[' + obj.value.map(function (v) { return v.toCSS(); }).join(', ') + ']';
    } else {
        return obj.toCSS();
    }
};

module.exports = jsEvalNode;

},{"./node.js":51,"./variable.js":63}],46:[function(require,module,exports){
var Node = require("./node.js");

var Keyword = function (value) { this.value = value; };
Keyword.prototype = new Node();
Keyword.prototype.type = "Keyword";
Keyword.prototype.genCSS = function (env, output) {
    if (this.value === '%') { throw { type: "Syntax", message: "Invalid % without number" }; }
    output.add(this.value);
};
Keyword.prototype.compare = function (other) {
    if (other instanceof Keyword) {
        return other.value === this.value ? 0 : 1;
    } else {
        return -1;
    }
};

Keyword.True = new(Keyword)('true');
Keyword.False = new(Keyword)('false');

module.exports = Keyword;

},{"./node.js":51}],47:[function(require,module,exports){
var Ruleset = require("./ruleset.js"),
    Value = require("./value.js"),
    Element = require("./element.js"),
    Selector = require("./selector.js"),
    Anonymous = require("./anonymous.js"),
    Expression = require("./expression.js"),
    Directive = require("./directive.js");

var Media = function (value, features, index, currentFileInfo) {
    this.index = index;
    this.currentFileInfo = currentFileInfo;

    var selectors = this.emptySelectors();

    this.features = new(Value)(features);
    this.rules = [new(Ruleset)(selectors, value)];
    this.rules[0].allowImports = true;
};
Media.prototype = new Directive();
Media.prototype.type = "Media";
Media.prototype.isRulesetLike = true;
Media.prototype.accept = function (visitor) {
    if (this.features) {
        this.features = visitor.visit(this.features);
    }
    if (this.rules) {
        this.rules = visitor.visitArray(this.rules);
    }
};
Media.prototype.genCSS = function (env, output) {
    output.add('@media ', this.currentFileInfo, this.index);
    this.features.genCSS(env, output);
    this.outputRuleset(env, output, this.rules);
};
Media.prototype.eval = function (env) {
    if (!env.mediaBlocks) {
        env.mediaBlocks = [];
        env.mediaPath = [];
    }

    var media = new(Media)(null, [], this.index, this.currentFileInfo);
    if(this.debugInfo) {
        this.rules[0].debugInfo = this.debugInfo;
        media.debugInfo = this.debugInfo;
    }
    var strictMathBypass = false;
    if (!env.strictMath) {
        strictMathBypass = true;
        env.strictMath = true;
    }
    try {
        media.features = this.features.eval(env);
    }
    finally {
        if (strictMathBypass) {
            env.strictMath = false;
        }
    }

    env.mediaPath.push(media);
    env.mediaBlocks.push(media);

    env.frames.unshift(this.rules[0]);
    media.rules = [this.rules[0].eval(env)];
    env.frames.shift();

    env.mediaPath.pop();

    return env.mediaPath.length === 0 ? media.evalTop(env) :
                media.evalNested(env);
};
//TODO merge with directive
Media.prototype.variable = function (name) { return Ruleset.prototype.variable.call(this.rules[0], name); };
Media.prototype.find = function () { return Ruleset.prototype.find.apply(this.rules[0], arguments); };
Media.prototype.rulesets = function () { return Ruleset.prototype.rulesets.apply(this.rules[0]); };
Media.prototype.emptySelectors = function() {
    var el = new(Element)('', '&', this.index, this.currentFileInfo),
        sels = [new(Selector)([el], null, null, this.index, this.currentFileInfo)];
    sels[0].mediaEmpty = true;
    return sels;
};
Media.prototype.markReferenced = function () {
    var i, rules = this.rules[0].rules;
    this.rules[0].markReferenced();
    this.isReferenced = true;
    for (i = 0; i < rules.length; i++) {
        if (rules[i].markReferenced) {
            rules[i].markReferenced();
        }
    }
};
Media.prototype.evalTop = function (env) {
    var result = this;

    // Render all dependent Media blocks.
    if (env.mediaBlocks.length > 1) {
        var selectors = this.emptySelectors();
        result = new(Ruleset)(selectors, env.mediaBlocks);
        result.multiMedia = true;
    }

    delete env.mediaBlocks;
    delete env.mediaPath;

    return result;
};
Media.prototype.evalNested = function (env) {
    var i, value,
        path = env.mediaPath.concat([this]);

    // Extract the media-query conditions separated with `,` (OR).
    for (i = 0; i < path.length; i++) {
        value = path[i].features instanceof Value ?
                    path[i].features.value : path[i].features;
        path[i] = Array.isArray(value) ? value : [value];
    }

    // Trace all permutations to generate the resulting media-query.
    //
    // (a, b and c) with nested (d, e) ->
    //    a and d
    //    a and e
    //    b and c and d
    //    b and c and e
    this.features = new(Value)(this.permute(path).map(function (path) {
        path = path.map(function (fragment) {
            return fragment.toCSS ? fragment : new(Anonymous)(fragment);
        });

        for(i = path.length - 1; i > 0; i--) {
            path.splice(i, 0, new(Anonymous)("and"));
        }

        return new(Expression)(path);
    }));

    // Fake a tree-node that doesn't output anything.
    return new(Ruleset)([], []);
};
Media.prototype.permute = function (arr) {
  if (arr.length === 0) {
      return [];
  } else if (arr.length === 1) {
      return arr[0];
  } else {
      var result = [];
      var rest = this.permute(arr.slice(1));
      for (var i = 0; i < rest.length; i++) {
          for (var j = 0; j < arr[0].length; j++) {
              result.push([arr[0][j]].concat(rest[i]));
          }
      }
      return result;
  }
};
Media.prototype.bubbleSelectors = function (selectors) {
  if (!selectors)
    return;
  this.rules = [new(Ruleset)(selectors.slice(0), [this.rules[0]])];
};
module.exports = Media;

},{"./anonymous.js":27,"./directive.js":38,"./element.js":39,"./expression.js":40,"./ruleset.js":57,"./selector.js":58,"./value.js":62}],48:[function(require,module,exports){
var Node = require("./node.js"),
    Selector = require("./selector.js"),
    MixinDefinition = require("./mixin-definition.js"),
    defaultFunc = require("../functions/default.js");

var MixinCall = function (elements, args, index, currentFileInfo, important) {
    this.selector = new(Selector)(elements);
    this.arguments = (args && args.length) ? args : null;
    this.index = index;
    this.currentFileInfo = currentFileInfo;
    this.important = important;
};
MixinCall.prototype = new Node();
MixinCall.prototype.type = "MixinCall";
MixinCall.prototype.accept = function (visitor) {
    if (this.selector) {
        this.selector = visitor.visit(this.selector);
    }
    if (this.arguments) {
        this.arguments = visitor.visitArray(this.arguments);
    }
};
MixinCall.prototype.eval = function (env) {
    var mixins, mixin, args, rules = [], match = false, i, m, f, isRecursive, isOneFound, rule,
        candidates = [], candidate, conditionResult = [],
        defaultResult, defNone = 0, defTrue = 1, defFalse = 2, count, originalRuleset;

    args = this.arguments && this.arguments.map(function (a) {
        return { name: a.name, value: a.value.eval(env) };
    });

    for (i = 0; i < env.frames.length; i++) {
        if ((mixins = env.frames[i].find(this.selector)).length > 0) {
            isOneFound = true;

            // To make `default()` function independent of definition order we have two "subpasses" here.
            // At first we evaluate each guard *twice* (with `default() == true` and `default() == false`),
            // and build candidate list with corresponding flags. Then, when we know all possible matches,
            // we make a final decision.

            for (m = 0; m < mixins.length; m++) {
                mixin = mixins[m];
                isRecursive = false;
                for(f = 0; f < env.frames.length; f++) {
                    if ((!(mixin instanceof MixinDefinition)) && mixin === (env.frames[f].originalRuleset || env.frames[f])) {
                        isRecursive = true;
                        break;
                    }
                }
                if (isRecursive) {
                    continue;
                }

                if (mixin.matchArgs(args, env)) {
                    candidate = {mixin: mixin, group: defNone};

                    if (mixin.matchCondition) {
                        for (f = 0; f < 2; f++) {
                            defaultFunc.value(f);
                            conditionResult[f] = mixin.matchCondition(args, env);
                        }
                        if (conditionResult[0] || conditionResult[1]) {
                            if (conditionResult[0] != conditionResult[1]) {
                                candidate.group = conditionResult[1] ?
                                    defTrue : defFalse;
                            }

                            candidates.push(candidate);
                        }
                    }
                    else {
                        candidates.push(candidate);
                    }

                    match = true;
                }
            }

            defaultFunc.reset();

            count = [0, 0, 0];
            for (m = 0; m < candidates.length; m++) {
                count[candidates[m].group]++;
            }

            if (count[defNone] > 0) {
                defaultResult = defFalse;
            } else {
                defaultResult = defTrue;
                if ((count[defTrue] + count[defFalse]) > 1) {
                    throw { type: 'Runtime',
                        message: 'Ambiguous use of `default()` found when matching for `'
                            + this.format(args) + '`',
                        index: this.index, filename: this.currentFileInfo.filename };
                }
            }

            for (m = 0; m < candidates.length; m++) {
                candidate = candidates[m].group;
                if ((candidate === defNone) || (candidate === defaultResult)) {
                    try {
                        mixin = candidates[m].mixin;
                        if (!(mixin instanceof MixinDefinition)) {
                            originalRuleset = mixin.originalRuleset || mixin;
                            mixin = new MixinDefinition("", [], mixin.rules, null, false);
                            mixin.originalRuleset = originalRuleset;
                        }
                        Array.prototype.push.apply(
                              rules, mixin.evalCall(env, args, this.important).rules);
                    } catch (e) {
                        throw { message: e.message, index: this.index, filename: this.currentFileInfo.filename, stack: e.stack };
                    }
                }
            }

            if (match) {
                if (!this.currentFileInfo || !this.currentFileInfo.reference) {
                    for (i = 0; i < rules.length; i++) {
                        rule = rules[i];
                        if (rule.markReferenced) {
                            rule.markReferenced();
                        }
                    }
                }
                return rules;
            }
        }
    }
    if (isOneFound) {
        throw { type:    'Runtime',
                message: 'No matching definition was found for `' + this.format(args) + '`',
                index:   this.index, filename: this.currentFileInfo.filename };
    } else {
        throw { type:    'Name',
                message: this.selector.toCSS().trim() + " is undefined",
                index:   this.index, filename: this.currentFileInfo.filename };
    }
};
MixinCall.prototype.format = function (args) {
    return this.selector.toCSS().trim() + '(' +
        (args ? args.map(function (a) {
            var argValue = "";
            if (a.name) {
                argValue += a.name + ":";
            }
            if (a.value.toCSS) {
                argValue += a.value.toCSS();
            } else {
                argValue += "???";
            }
            return argValue;
        }).join(', ') : "") + ")";
};
module.exports = MixinCall;

},{"../functions/default.js":10,"./mixin-definition.js":49,"./node.js":51,"./selector.js":58}],49:[function(require,module,exports){
var Selector = require("./selector.js"),
    Element = require("./element.js"),
    Ruleset = require("./ruleset.js"),
    Rule = require("./rule.js"),
    Expression = require("./expression.js"),
    contexts = require("../contexts.js");

var Definition = function (name, params, rules, condition, variadic, frames) {
    this.name = name;
    this.selectors = [new(Selector)([new(Element)(null, name, this.index, this.currentFileInfo)])];
    this.params = params;
    this.condition = condition;
    this.variadic = variadic;
    this.arity = params.length;
    this.rules = rules;
    this._lookups = {};
    this.required = params.reduce(function (count, p) {
        if (!p.name || (p.name && !p.value)) { return count + 1; }
        else                                 { return count; }
    }, 0);
    this.frames = frames;
};
Definition.prototype = new Ruleset();
Definition.prototype.type = "MixinDefinition";
Definition.prototype.evalFirst = true;
Definition.prototype.accept = function (visitor) {
    if (this.params && this.params.length) {
        this.params = visitor.visitArray(this.params);
    }
    this.rules = visitor.visitArray(this.rules);
    if (this.condition) {
        this.condition = visitor.visit(this.condition);
    }
};
Definition.prototype.evalParams = function (env, mixinEnv, args, evaldArguments) {
    /*jshint boss:true */
    var frame = new(Ruleset)(null, null),
        varargs, arg,
        params = this.params.slice(0),
        i, j, val, name, isNamedFound, argIndex, argsLength = 0;

    mixinEnv = new contexts.evalEnv(mixinEnv, [frame].concat(mixinEnv.frames));

    if (args) {
        args = args.slice(0);
        argsLength = args.length;

        for(i = 0; i < argsLength; i++) {
            arg = args[i];
            if (name = (arg && arg.name)) {
                isNamedFound = false;
                for(j = 0; j < params.length; j++) {
                    if (!evaldArguments[j] && name === params[j].name) {
                        evaldArguments[j] = arg.value.eval(env);
                        frame.prependRule(new(Rule)(name, arg.value.eval(env)));
                        isNamedFound = true;
                        break;
                    }
                }
                if (isNamedFound) {
                    args.splice(i, 1);
                    i--;
                    continue;
                } else {
                    throw { type: 'Runtime', message: "Named argument for " + this.name +
                        ' ' + args[i].name + ' not found' };
                }
            }
        }
    }
    argIndex = 0;
    for (i = 0; i < params.length; i++) {
        if (evaldArguments[i]) { continue; }

        arg = args && args[argIndex];

        if (name = params[i].name) {
            if (params[i].variadic) {
                varargs = [];
                for (j = argIndex; j < argsLength; j++) {
                    varargs.push(args[j].value.eval(env));
                }
                frame.prependRule(new(Rule)(name, new(Expression)(varargs).eval(env)));
            } else {
                val = arg && arg.value;
                if (val) {
                    val = val.eval(env);
                } else if (params[i].value) {
                    val = params[i].value.eval(mixinEnv);
                    frame.resetCache();
                } else {
                    throw { type: 'Runtime', message: "wrong number of arguments for " + this.name +
                        ' (' + argsLength + ' for ' + this.arity + ')' };
                }

                frame.prependRule(new(Rule)(name, val));
                evaldArguments[i] = val;
            }
        }

        if (params[i].variadic && args) {
            for (j = argIndex; j < argsLength; j++) {
                evaldArguments[j] = args[j].value.eval(env);
            }
        }
        argIndex++;
    }

    return frame;
};
Definition.prototype.eval = function (env) {
    return new Definition(this.name, this.params, this.rules, this.condition, this.variadic, this.frames || env.frames.slice(0));
};
Definition.prototype.evalCall = function (env, args, important) {
    var _arguments = [],
        mixinFrames = this.frames ? this.frames.concat(env.frames) : env.frames,
        frame = this.evalParams(env, new(contexts.evalEnv)(env, mixinFrames), args, _arguments),
        rules, ruleset;

    frame.prependRule(new(Rule)('@arguments', new(Expression)(_arguments).eval(env)));

    rules = this.rules.slice(0);

    ruleset = new(Ruleset)(null, rules);
    ruleset.originalRuleset = this;
    ruleset = ruleset.eval(new(contexts.evalEnv)(env, [this, frame].concat(mixinFrames)));
    if (important) {
        ruleset = this.makeImportant.apply(ruleset);
    }
    return ruleset;
};
Definition.prototype.matchCondition = function (args, env) {
    if (this.condition && !this.condition.eval(
        new(contexts.evalEnv)(env,
            [this.evalParams(env, new(contexts.evalEnv)(env, this.frames ? this.frames.concat(env.frames) : env.frames), args, [])] // the parameter variables
                .concat(this.frames) // the parent namespace/mixin frames
                .concat(env.frames)))) { // the current environment frames
        return false;
    }
    return true;
};
Definition.prototype.matchArgs = function (args, env) {
    var argsLength = (args && args.length) || 0, len;

    if (! this.variadic) {
        if (argsLength < this.required)                               { return false; }
        if (argsLength > this.params.length)                          { return false; }
    } else {
        if (argsLength < (this.required - 1))                         { return false; }
    }

    len = Math.min(argsLength, this.arity);

    for (var i = 0; i < len; i++) {
        if (!this.params[i].name && !this.params[i].variadic) {
            if (args[i].value.eval(env).toCSS() != this.params[i].value.eval(env).toCSS()) {
                return false;
            }
        }
    }
    return true;
};
module.exports = Definition;

},{"../contexts.js":2,"./element.js":39,"./expression.js":40,"./rule.js":55,"./ruleset.js":57,"./selector.js":58}],50:[function(require,module,exports){
var Node = require("./node.js"),
    Operation = require("./operation.js"),
    Dimension = require("./dimension.js");

var Negative = function (node) {
    this.value = node;
};
Negative.prototype = new Node();
Negative.prototype.type = "Negative";
Negative.prototype.genCSS = function (env, output) {
    output.add('-');
    this.value.genCSS(env, output);
};
Negative.prototype.eval = function (env) {
    if (env.isMathOn()) {
        return (new(Operation)('*', [new(Dimension)(-1), this.value])).eval(env);
    }
    return new(Negative)(this.value.eval(env));
};
module.exports = Negative;

},{"./dimension.js":37,"./node.js":51,"./operation.js":52}],51:[function(require,module,exports){
var Node = function() {
};
Node.prototype.toCSS = function (env) {
    var strs = [];
    this.genCSS(env, {
        add: function(chunk, fileInfo, index) {
            strs.push(chunk);
        },
        isEmpty: function () {
            return strs.length === 0;
        }
    });
    return strs.join('');
};
Node.prototype.genCSS = function (env, output) {
    output.add(this.value);
};
Node.prototype.accept = function (visitor) {
    this.value = visitor.visit(this.value);
};
Node.prototype.eval = function () { return this; };
Node.prototype._operate = function (env, op, a, b) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return a / b;
    }
};
Node.prototype.fround = function(env, value) {
    var precision = env && env.numPrecision;
    //add "epsilon" to ensure numbers like 1.000000005 (represented as 1.000000004999....) are properly rounded...
    return (precision == null) ? value : Number((value + 2e-16).toFixed(precision));
};
module.exports = Node;

},{}],52:[function(require,module,exports){
var Node = require("./node.js"),
    Color = require("./color.js"),
    Dimension = require("./dimension.js");

var Operation = function (op, operands, isSpaced) {
    this.op = op.trim();
    this.operands = operands;
    this.isSpaced = isSpaced;
};
Operation.prototype = new Node();
Operation.prototype.type = "Operation";
Operation.prototype.accept = function (visitor) {
    this.operands = visitor.visit(this.operands);
};
Operation.prototype.eval = function (env) {
    var a = this.operands[0].eval(env),
        b = this.operands[1].eval(env);

    if (env.isMathOn()) {
        if (a instanceof Dimension && b instanceof Color) {
            a = a.toColor();
        }
        if (b instanceof Dimension && a instanceof Color) {
            b = b.toColor();
        }
        if (!a.operate) {
            throw { type: "Operation",
                    message: "Operation on an invalid type" };
        }

        return a.operate(env, this.op, b);
    } else {
        return new(Operation)(this.op, [a, b], this.isSpaced);
    }
};
Operation.prototype.genCSS = function (env, output) {
    this.operands[0].genCSS(env, output);
    if (this.isSpaced) {
        output.add(" ");
    }
    output.add(this.op);
    if (this.isSpaced) {
        output.add(" ");
    }
    this.operands[1].genCSS(env, output);
};

module.exports = Operation;

},{"./color.js":31,"./dimension.js":37,"./node.js":51}],53:[function(require,module,exports){
var Node = require("./node.js");

var Paren = function (node) {
    this.value = node;
};
Paren.prototype = new Node();
Paren.prototype.type = "Paren";
Paren.prototype.genCSS = function (env, output) {
    output.add('(');
    this.value.genCSS(env, output);
    output.add(')');
};
Paren.prototype.eval = function (env) {
    return new(Paren)(this.value.eval(env));
};
module.exports = Paren;

},{"./node.js":51}],54:[function(require,module,exports){
var JsEvalNode = require("./js-eval-node.js"),
    Variable = require("./variable.js");

var Quoted = function (str, content, escaped, index, currentFileInfo) {
    this.escaped = escaped;
    this.value = content || '';
    this.quote = str.charAt(0);
    this.index = index;
    this.currentFileInfo = currentFileInfo;
};
Quoted.prototype = new JsEvalNode();
Quoted.prototype.type = "Quoted";
Quoted.prototype.genCSS = function (env, output) {
    if (!this.escaped) {
        output.add(this.quote, this.currentFileInfo, this.index);
    }
    output.add(this.value);
    if (!this.escaped) {
        output.add(this.quote);
    }
};
Quoted.prototype.eval = function (env) {
    var that = this;
    var value = this.value.replace(/`([^`]+)`/g, function (_, exp) {
        return String(that.evaluateJavaScript(exp, env));
    }).replace(/@\{([\w-]+)\}/g, function (_, name) {
        var v = new(Variable)('@' + name, that.index, that.currentFileInfo).eval(env, true);
        return (v instanceof Quoted) ? v.value : v.toCSS();
    });
    return new(Quoted)(this.quote + value + this.quote, value, this.escaped, this.index, this.currentFileInfo);
};
Quoted.prototype.compare = function (x) {
    if (!x.toCSS) {
        return -1;
    }

    var left, right;

    // when comparing quoted strings allow the quote to differ
    if (x.type === "Quoted" && !this.escaped && !x.escaped) {
        left = x.value;
        right = this.value;
    } else {
        left = this.toCSS();
        right = x.toCSS();
    }

    if (left === right) {
        return 0;
    }

    return left < right ? -1 : 1;
};
module.exports = Quoted;

},{"./js-eval-node.js":45,"./variable.js":63}],55:[function(require,module,exports){
var Node = require("./node.js"),
    Value = require("./value.js"),
    Keyword = require("./keyword.js");

var Rule = function (name, value, important, merge, index, currentFileInfo, inline, variable) {
    this.name = name;
    this.value = (value instanceof Node) ? value : new(Value)([value]); //value instanceof tree.Value || value instanceof tree.Ruleset ??
    this.important = important ? ' ' + important.trim() : '';
    this.merge = merge;
    this.index = index;
    this.currentFileInfo = currentFileInfo;
    this.inline = inline || false;
    this.variable = (variable !== undefined) ? variable
        : (name.charAt && (name.charAt(0) === '@'));
};

function evalName(env, name) {
    var value = "", i, n = name.length,
        output = {add: function (s) {value += s;}};
    for (i = 0; i < n; i++) {
        name[i].eval(env).genCSS(env, output);
    }
    return value;
}

Rule.prototype = new Node();
Rule.prototype.type = "Rule";
Rule.prototype.genCSS = function (env, output) {
    output.add(this.name + (env.compress ? ':' : ': '), this.currentFileInfo, this.index);
    try {
        this.value.genCSS(env, output);
    }
    catch(e) {
        e.index = this.index;
        e.filename = this.currentFileInfo.filename;
        throw e;
    }
    output.add(this.important + ((this.inline || (env.lastRule && env.compress)) ? "" : ";"), this.currentFileInfo, this.index);
};
Rule.prototype.eval = function (env) {
    var strictMathBypass = false, name = this.name, evaldValue, variable = this.variable;
    if (typeof name !== "string") {
        // expand 'primitive' name directly to get
        // things faster (~10% for benchmark.less):
        name = (name.length === 1)
            && (name[0] instanceof Keyword)
                ? name[0].value : evalName(env, name);
            variable = false; // never treat expanded interpolation as new variable name
    }
    if (name === "font" && !env.strictMath) {
        strictMathBypass = true;
        env.strictMath = true;
    }
    try {
        evaldValue = this.value.eval(env);

        if (!this.variable && evaldValue.type === "DetachedRuleset") {
            throw { message: "Rulesets cannot be evaluated on a property.",
                    index: this.index, filename: this.currentFileInfo.filename };
        }

        return new(Rule)(name,
                          evaldValue,
                          this.important,
                          this.merge,
                          this.index, this.currentFileInfo, this.inline,
                              variable);
    }
    catch(e) {
        if (typeof e.index !== 'number') {
            e.index = this.index;
            e.filename = this.currentFileInfo.filename;
        }
        throw e;
    }
    finally {
        if (strictMathBypass) {
            env.strictMath = false;
        }
    }
};
Rule.prototype.makeImportant = function () {
    return new(Rule)(this.name,
                          this.value,
                          "!important",
                          this.merge,
                          this.index, this.currentFileInfo, this.inline);
};

module.exports = Rule;

},{"./keyword.js":46,"./node.js":51,"./value.js":62}],56:[function(require,module,exports){
var Node = require("./node.js"),
    Variable = require("./variable.js");

var RulesetCall = function (variable) {
    this.variable = variable;
};
RulesetCall.prototype = new Node();
RulesetCall.prototype.type = "RulesetCall";
RulesetCall.prototype.eval = function (env) {
    var detachedRuleset = new(Variable)(this.variable).eval(env);
    return detachedRuleset.callEval(env);
};
module.exports = RulesetCall;

},{"./node.js":51,"./variable.js":63}],57:[function(require,module,exports){
var Node = require("./node.js"),
    Rule = require("./rule.js"),
    Selector = require("./selector.js"),
    Element = require("./element.js"),
    contexts = require("../contexts.js"),
    defaultFunc = require("../functions/default.js"),
    getDebugInfo = require("./debug-info.js");

var Ruleset = function (selectors, rules, strictImports) {
    this.selectors = selectors;
    this.rules = rules;
    this._lookups = {};
    this.strictImports = strictImports;
};
Ruleset.prototype = new Node();
Ruleset.prototype.type = "Ruleset";
Ruleset.prototype.isRuleset = true;
Ruleset.prototype.isRulesetLike = true;
Ruleset.prototype.accept = function (visitor) {
    if (this.paths) {
        visitor.visitArray(this.paths, true);
    } else if (this.selectors) {
        this.selectors = visitor.visitArray(this.selectors);
    }
    if (this.rules && this.rules.length) {
        this.rules = visitor.visitArray(this.rules);
    }
};
Ruleset.prototype.eval = function (env) {
    var thisSelectors = this.selectors, selectors,
        selCnt, selector, i, hasOnePassingSelector = false;

    if (thisSelectors && (selCnt = thisSelectors.length)) {
        selectors = [];
        defaultFunc.error({
            type: "Syntax",
            message: "it is currently only allowed in parametric mixin guards,"
        });
        for (i = 0; i < selCnt; i++) {
            selector = thisSelectors[i].eval(env);
            selectors.push(selector);
            if (selector.evaldCondition) {
                hasOnePassingSelector = true;
            }
        }
        defaultFunc.reset();
    } else {
        hasOnePassingSelector = true;
    }

    var rules = this.rules ? this.rules.slice(0) : null,
        ruleset = new(Ruleset)(selectors, rules, this.strictImports),
        rule, subRule;

    ruleset.originalRuleset = this;
    ruleset.root = this.root;
    ruleset.firstRoot = this.firstRoot;
    ruleset.allowImports = this.allowImports;

    if(this.debugInfo) {
        ruleset.debugInfo = this.debugInfo;
    }

    if (!hasOnePassingSelector) {
        rules.length = 0;
    }

    // push the current ruleset to the frames stack
    var envFrames = env.frames;
    envFrames.unshift(ruleset);

    // currrent selectors
    var envSelectors = env.selectors;
    if (!envSelectors) {
        env.selectors = envSelectors = [];
    }
    envSelectors.unshift(this.selectors);

    // Evaluate imports
    if (ruleset.root || ruleset.allowImports || !ruleset.strictImports) {
        ruleset.evalImports(env);
    }

    // Store the frames around mixin definitions,
    // so they can be evaluated like closures when the time comes.
    var rsRules = ruleset.rules, rsRuleCnt = rsRules ? rsRules.length : 0;
    for (i = 0; i < rsRuleCnt; i++) {
        if (rsRules[i].evalFirst) {
            rsRules[i] = rsRules[i].eval(env);
        }
    }

    var mediaBlockCount = (env.mediaBlocks && env.mediaBlocks.length) || 0;

    // Evaluate mixin calls.
    for (i = 0; i < rsRuleCnt; i++) {
        if (rsRules[i].type === "MixinCall") {
            /*jshint loopfunc:true */
            rules = rsRules[i].eval(env).filter(function(r) {
                if ((r instanceof Rule) && r.variable) {
                    // do not pollute the scope if the variable is
                    // already there. consider returning false here
                    // but we need a way to "return" variable from mixins
                    return !(ruleset.variable(r.name));
                }
                return true;
            });
            rsRules.splice.apply(rsRules, [i, 1].concat(rules));
            rsRuleCnt += rules.length - 1;
            i += rules.length-1;
            ruleset.resetCache();
        } else if (rsRules[i].type === "RulesetCall") {
            /*jshint loopfunc:true */
            rules = rsRules[i].eval(env).rules.filter(function(r) {
                if ((r instanceof Rule) && r.variable) {
                    // do not pollute the scope at all
                    return false;
                }
                return true;
            });
            rsRules.splice.apply(rsRules, [i, 1].concat(rules));
            rsRuleCnt += rules.length - 1;
            i += rules.length-1;
            ruleset.resetCache();
        }
    }

    // Evaluate everything else
    for (i = 0; i < rsRules.length; i++) {
        rule = rsRules[i];
        if (!rule.evalFirst) {
            rsRules[i] = rule = rule.eval ? rule.eval(env) : rule;
        }
    }

    // Evaluate everything else
    for (i = 0; i < rsRules.length; i++) {
        rule = rsRules[i];
        // for rulesets, check if it is a css guard and can be removed
        if (rule instanceof Ruleset && rule.selectors && rule.selectors.length === 1) {
            // check if it can be folded in (e.g. & where)
            if (rule.selectors[0].isJustParentSelector()) {
                rsRules.splice(i--, 1);

                for(var j = 0; j < rule.rules.length; j++) {
                    subRule = rule.rules[j];
                    if (!(subRule instanceof Rule) || !subRule.variable) {
                        rsRules.splice(++i, 0, subRule);
                    }
                }
            }
        }
    }

    // Pop the stack
    envFrames.shift();
    envSelectors.shift();

    if (env.mediaBlocks) {
        for (i = mediaBlockCount; i < env.mediaBlocks.length; i++) {
            env.mediaBlocks[i].bubbleSelectors(selectors);
        }
    }

    return ruleset;
};
Ruleset.prototype.evalImports = function(env) {
    var rules = this.rules, i, importRules;
    if (!rules) { return; }

    for (i = 0; i < rules.length; i++) {
        if (rules[i].type === "Import") {
            importRules = rules[i].eval(env);
            if (importRules && importRules.length) {
                rules.splice.apply(rules, [i, 1].concat(importRules));
                i+= importRules.length-1;
            } else {
                rules.splice(i, 1, importRules);
            }
            this.resetCache();
        }
    }
};
Ruleset.prototype.makeImportant = function() {
    return new Ruleset(this.selectors, this.rules.map(function (r) {
                if (r.makeImportant) {
                    return r.makeImportant();
                } else {
                    return r;
                }
            }), this.strictImports);
};
Ruleset.prototype.matchArgs = function (args) {
    return !args || args.length === 0;
};
// lets you call a css selector with a guard
Ruleset.prototype.matchCondition = function (args, env) {
    var lastSelector = this.selectors[this.selectors.length-1];
    if (!lastSelector.evaldCondition) {
        return false;
    }
    if (lastSelector.condition &&
        !lastSelector.condition.eval(
            new(contexts.evalEnv)(env,
                env.frames))) {
        return false;
    }
    return true;
};
Ruleset.prototype.resetCache = function () {
    this._rulesets = null;
    this._variables = null;
    this._lookups = {};
};
Ruleset.prototype.variables = function () {
    if (!this._variables) {
        this._variables = !this.rules ? {} : this.rules.reduce(function (hash, r) {
            if (r instanceof Rule && r.variable === true) {
                hash[r.name] = r;
            }
            return hash;
        }, {});
    }
    return this._variables;
};
Ruleset.prototype.variable = function (name) {
    return this.variables()[name];
};
Ruleset.prototype.rulesets = function () {
    if (!this.rules) { return null; }

    var filtRules = [], rules = this.rules, cnt = rules.length,
        i, rule;

    for (i = 0; i < cnt; i++) {
        rule = rules[i];
        if (rule.isRuleset) {
            filtRules.push(rule);
        }
    }

    return filtRules;
};
Ruleset.prototype.prependRule = function (rule) {
    var rules = this.rules;
    if (rules) { rules.unshift(rule); } else { this.rules = [ rule ]; }
};
Ruleset.prototype.find = function (selector, self) {
    self = self || this;
    var rules = [], match,
        key = selector.toCSS();

    if (key in this._lookups) { return this._lookups[key]; }

    this.rulesets().forEach(function (rule) {
        if (rule !== self) {
            for (var j = 0; j < rule.selectors.length; j++) {
                match = selector.match(rule.selectors[j]);
                if (match) {
                    if (selector.elements.length > match) {
                        Array.prototype.push.apply(rules, rule.find(
                            new(Selector)(selector.elements.slice(match)), self));
                    } else {
                        rules.push(rule);
                    }
                    break;
                }
            }
        }
    });
    this._lookups[key] = rules;
    return rules;
};
Ruleset.prototype.genCSS = function (env, output) {
    var i, j,
        charsetRuleNodes = [],
        ruleNodes = [],
        rulesetNodes = [],
        rulesetNodeCnt,
        debugInfo,     // Line number debugging
        rule,
        path;

    env.tabLevel = (env.tabLevel || 0);

    if (!this.root) {
        env.tabLevel++;
    }

    var tabRuleStr = env.compress ? '' : Array(env.tabLevel + 1).join("  "),
        tabSetStr = env.compress ? '' : Array(env.tabLevel).join("  "),
        sep;

    function isRulesetLikeNode(rule, root) {
         // if it has nested rules, then it should be treated like a ruleset
         // medias and comments do not have nested rules, but should be treated like rulesets anyway
         // some directives and anonymous nodes are ruleset like, others are not
         if (typeof rule.isRulesetLike === "boolean")
         {
             return rule.isRulesetLike;
         } else if (typeof rule.isRulesetLike === "function")
         {
             return rule.isRulesetLike(root);
         }

         //anything else is assumed to be a rule
         return false;
    }

    for (i = 0; i < this.rules.length; i++) {
        rule = this.rules[i];
        if (isRulesetLikeNode(rule, this.root)) {
            rulesetNodes.push(rule);
        } else {
            //charsets should float on top of everything
            if (rule.isCharset && rule.isCharset()) {
                charsetRuleNodes.push(rule);
            } else {
                ruleNodes.push(rule);
            }
        }
    }
    ruleNodes = charsetRuleNodes.concat(ruleNodes);

    // If this is the root node, we don't render
    // a selector, or {}.
    if (!this.root) {
        debugInfo = getDebugInfo(env, this, tabSetStr);

        if (debugInfo) {
            output.add(debugInfo);
            output.add(tabSetStr);
        }

        var paths = this.paths, pathCnt = paths.length,
            pathSubCnt;

        sep = env.compress ? ',' : (',\n' + tabSetStr);

        for (i = 0; i < pathCnt; i++) {
            path = paths[i];
            if (!(pathSubCnt = path.length)) { continue; }
            if (i > 0) { output.add(sep); }

            env.firstSelector = true;
            path[0].genCSS(env, output);

            env.firstSelector = false;
            for (j = 1; j < pathSubCnt; j++) {
                path[j].genCSS(env, output);
            }
        }

        output.add((env.compress ? '{' : ' {\n') + tabRuleStr);
    }

    // Compile rules and rulesets
    for (i = 0; i < ruleNodes.length; i++) {
        rule = ruleNodes[i];

        // @page{ directive ends up with root elements inside it, a mix of rules and rulesets
        // In this instance we do not know whether it is the last property
        if (i + 1 === ruleNodes.length && (!this.root || rulesetNodes.length === 0 || this.firstRoot)) {
            env.lastRule = true;
        }

        if (rule.genCSS) {
            rule.genCSS(env, output);
        } else if (rule.value) {
            output.add(rule.value.toString());
        }

        if (!env.lastRule) {
            output.add(env.compress ? '' : ('\n' + tabRuleStr));
        } else {
            env.lastRule = false;
        }
    }

    if (!this.root) {
        output.add((env.compress ? '}' : '\n' + tabSetStr + '}'));
        env.tabLevel--;
    }

    sep = (env.compress ? "" : "\n") + (this.root ? tabRuleStr : tabSetStr);
    rulesetNodeCnt = rulesetNodes.length;
    if (rulesetNodeCnt) {
        if (ruleNodes.length && sep) { output.add(sep); }
        rulesetNodes[0].genCSS(env, output);
        for (i = 1; i < rulesetNodeCnt; i++) {
            if (sep) { output.add(sep); }
            rulesetNodes[i].genCSS(env, output);
        }
    }

    if (!output.isEmpty() && !env.compress && this.firstRoot) {
        output.add('\n');
    }
};
Ruleset.prototype.markReferenced = function () {
    if (!this.selectors) {
        return;
    }
    for (var s = 0; s < this.selectors.length; s++) {
        this.selectors[s].markReferenced();
    }
};
Ruleset.prototype.joinSelectors = function (paths, context, selectors) {
    for (var s = 0; s < selectors.length; s++) {
        this.joinSelector(paths, context, selectors[s]);
    }
};
Ruleset.prototype.joinSelector = function (paths, context, selector) {

    var i, j, k,
        hasParentSelector, newSelectors, el, sel, parentSel,
        newSelectorPath, afterParentJoin, newJoinedSelector,
        newJoinedSelectorEmpty, lastSelector, currentElements,
        selectorsMultiplied;

    for (i = 0; i < selector.elements.length; i++) {
        el = selector.elements[i];
        if (el.value === '&') {
            hasParentSelector = true;
        }
    }

    if (!hasParentSelector) {
        if (context.length > 0) {
            for (i = 0; i < context.length; i++) {
                paths.push(context[i].concat(selector));
            }
        }
        else {
            paths.push([selector]);
        }
        return;
    }

    // The paths are [[Selector]]
    // The first list is a list of comma seperated selectors
    // The inner list is a list of inheritance seperated selectors
    // e.g.
    // .a, .b {
    //   .c {
    //   }
    // }
    // == [[.a] [.c]] [[.b] [.c]]
    //

    // the elements from the current selector so far
    currentElements = [];
    // the current list of new selectors to add to the path.
    // We will build it up. We initiate it with one empty selector as we "multiply" the new selectors
    // by the parents
    newSelectors = [[]];

    for (i = 0; i < selector.elements.length; i++) {
        el = selector.elements[i];
        // non parent reference elements just get added
        if (el.value !== "&") {
            currentElements.push(el);
        } else {
            // the new list of selectors to add
            selectorsMultiplied = [];

            // merge the current list of non parent selector elements
            // on to the current list of selectors to add
            if (currentElements.length > 0) {
                this.mergeElementsOnToSelectors(currentElements, newSelectors);
            }

            // loop through our current selectors
            for (j = 0; j < newSelectors.length; j++) {
                sel = newSelectors[j];
                // if we don't have any parent paths, the & might be in a mixin so that it can be used
                // whether there are parents or not
                if (context.length === 0) {
                    // the combinator used on el should now be applied to the next element instead so that
                    // it is not lost
                    if (sel.length > 0) {
                        sel[0].elements = sel[0].elements.slice(0);
                        sel[0].elements.push(new(Element)(el.combinator, '', el.index, el.currentFileInfo));
                    }
                    selectorsMultiplied.push(sel);
                }
                else {
                    // and the parent selectors
                    for (k = 0; k < context.length; k++) {
                        parentSel = context[k];
                        // We need to put the current selectors
                        // then join the last selector's elements on to the parents selectors

                        // our new selector path
                        newSelectorPath = [];
                        // selectors from the parent after the join
                        afterParentJoin = [];
                        newJoinedSelectorEmpty = true;

                        //construct the joined selector - if & is the first thing this will be empty,
                        // if not newJoinedSelector will be the last set of elements in the selector
                        if (sel.length > 0) {
                            newSelectorPath = sel.slice(0);
                            lastSelector = newSelectorPath.pop();
                            newJoinedSelector = selector.createDerived(lastSelector.elements.slice(0));
                            newJoinedSelectorEmpty = false;
                        }
                        else {
                            newJoinedSelector = selector.createDerived([]);
                        }

                        //put together the parent selectors after the join
                        if (parentSel.length > 1) {
                            afterParentJoin = afterParentJoin.concat(parentSel.slice(1));
                        }

                        if (parentSel.length > 0) {
                            newJoinedSelectorEmpty = false;

                            // join the elements so far with the first part of the parent
                            newJoinedSelector.elements.push(new(Element)(el.combinator, parentSel[0].elements[0].value, el.index, el.currentFileInfo));
                            newJoinedSelector.elements = newJoinedSelector.elements.concat(parentSel[0].elements.slice(1));
                        }

                        if (!newJoinedSelectorEmpty) {
                            // now add the joined selector
                            newSelectorPath.push(newJoinedSelector);
                        }

                        // and the rest of the parent
                        newSelectorPath = newSelectorPath.concat(afterParentJoin);

                        // add that to our new set of selectors
                        selectorsMultiplied.push(newSelectorPath);
                    }
                }
            }

            // our new selectors has been multiplied, so reset the state
            newSelectors = selectorsMultiplied;
            currentElements = [];
        }
    }

    // if we have any elements left over (e.g. .a& .b == .b)
    // add them on to all the current selectors
    if (currentElements.length > 0) {
        this.mergeElementsOnToSelectors(currentElements, newSelectors);
    }

    for (i = 0; i < newSelectors.length; i++) {
        if (newSelectors[i].length > 0) {
            paths.push(newSelectors[i]);
        }
    }
};
Ruleset.prototype.mergeElementsOnToSelectors = function(elements, selectors) {
    var i, sel;

    if (selectors.length === 0) {
        selectors.push([ new(Selector)(elements) ]);
        return;
    }

    for (i = 0; i < selectors.length; i++) {
        sel = selectors[i];

        // if the previous thing in sel is a parent this needs to join on to it
        if (sel.length > 0) {
            sel[sel.length - 1] = sel[sel.length - 1].createDerived(sel[sel.length - 1].elements.concat(elements));
        }
        else {
            sel.push(new(Selector)(elements));
        }
    }
};
module.exports = Ruleset;

},{"../contexts.js":2,"../functions/default.js":10,"./debug-info.js":35,"./element.js":39,"./node.js":51,"./rule.js":55,"./selector.js":58}],58:[function(require,module,exports){
var Node = require("./node.js");

var Selector = function (elements, extendList, condition, index, currentFileInfo, isReferenced) {
    this.elements = elements;
    this.extendList = extendList;
    this.condition = condition;
    this.currentFileInfo = currentFileInfo || {};
    this.isReferenced = isReferenced;
    if (!condition) {
        this.evaldCondition = true;
    }
};
Selector.prototype = new Node();
Selector.prototype.type = "Selector";
Selector.prototype.accept = function (visitor) {
    if (this.elements) {
        this.elements = visitor.visitArray(this.elements);
    }
    if (this.extendList) {
        this.extendList = visitor.visitArray(this.extendList);
    }
    if (this.condition) {
        this.condition = visitor.visit(this.condition);
    }
};
Selector.prototype.createDerived = function(elements, extendList, evaldCondition) {
    evaldCondition = (evaldCondition != null) ? evaldCondition : this.evaldCondition;
    var newSelector = new(Selector)(elements, extendList || this.extendList, null, this.index, this.currentFileInfo, this.isReferenced);
    newSelector.evaldCondition = evaldCondition;
    newSelector.mediaEmpty = this.mediaEmpty;
    return newSelector;
};
Selector.prototype.match = function (other) {
    var elements = this.elements,
        len = elements.length,
        olen, i;

    other.CacheElements();

    olen = other._elements.length;
    if (olen === 0 || len < olen) {
        return 0;
    } else {
        for (i = 0; i < olen; i++) {
            if (elements[i].value !== other._elements[i]) {
                return 0;
            }
        }
    }

    return olen; // return number of matched elements
};
Selector.prototype.CacheElements = function(){
    var css = '', len, v, i;

    if( !this._elements ){

        len = this.elements.length;
        for(i = 0; i < len; i++){

            v = this.elements[i];
            css += v.combinator.value;

            if( !v.value.value ){
                css += v.value;
                continue;
            }

            if( typeof v.value.value !== "string" ){
                css = '';
                break;
            }
            css += v.value.value;
        }

        this._elements = css.match(/[,&#\*\.\w-]([\w-]|(\\.))*/g);

        if (this._elements) {
            if (this._elements[0] === "&") {
                this._elements.shift();
            }

        } else {
            this._elements = [];
        }

    }
};
Selector.prototype.isJustParentSelector = function() {
    return !this.mediaEmpty &&
        this.elements.length === 1 &&
        this.elements[0].value === '&' &&
        (this.elements[0].combinator.value === ' ' || this.elements[0].combinator.value === '');
};
Selector.prototype.eval = function (env) {
    var evaldCondition = this.condition && this.condition.eval(env),
        elements = this.elements, extendList = this.extendList;

    elements = elements && elements.map(function (e) { return e.eval(env); });
    extendList = extendList && extendList.map(function(extend) { return extend.eval(env); });

    return this.createDerived(elements, extendList, evaldCondition);
};
Selector.prototype.genCSS = function (env, output) {
    var i, element;
    if ((!env || !env.firstSelector) && this.elements[0].combinator.value === "") {
        output.add(' ', this.currentFileInfo, this.index);
    }
    if (!this._css) {
        //TODO caching? speed comparison?
        for(i = 0; i < this.elements.length; i++) {
            element = this.elements[i];
            element.genCSS(env, output);
        }
    }
};
Selector.prototype.markReferenced = function () {
    this.isReferenced = true;
};
Selector.prototype.getIsReferenced = function() {
    return !this.currentFileInfo.reference || this.isReferenced;
};
Selector.prototype.getIsOutput = function() {
    return this.evaldCondition;
};
module.exports = Selector;

},{"./node.js":51}],59:[function(require,module,exports){
var Node = require("./node.js");

var UnicodeDescriptor = function (value) {
    this.value = value;
};
UnicodeDescriptor.prototype = new Node();
UnicodeDescriptor.prototype.type = "UnicodeDescriptor";

module.exports = UnicodeDescriptor;

},{"./node.js":51}],60:[function(require,module,exports){
var Node = require("./node.js"),
    unitConversions = require("../data/unit-conversions.js");

var Unit = function (numerator, denominator, backupUnit) {
    this.numerator = numerator ? numerator.slice(0).sort() : [];
    this.denominator = denominator ? denominator.slice(0).sort() : [];
    this.backupUnit = backupUnit;
};

Unit.prototype = new Node();
Unit.prototype.type = "Unit";
Unit.prototype.clone = function () {
    return new Unit(this.numerator.slice(0), this.denominator.slice(0), this.backupUnit);
};
Unit.prototype.genCSS = function (env, output) {
    if (this.numerator.length >= 1) {
        output.add(this.numerator[0]);
    } else
    if (this.denominator.length >= 1) {
        output.add(this.denominator[0]);
    } else
    if ((!env || !env.strictUnits) && this.backupUnit) {
        output.add(this.backupUnit);
    }
};
Unit.prototype.toString = function () {
    var i, returnStr = this.numerator.join("*");
    for (i = 0; i < this.denominator.length; i++) {
        returnStr += "/" + this.denominator[i];
    }
    return returnStr;
};
Unit.prototype.compare = function (other) {
    return this.is(other.toString()) ? 0 : -1;
};
Unit.prototype.is = function (unitString) {
    return this.toString() === unitString;
};
Unit.prototype.isLength = function () {
    return Boolean(this.toCSS().match(/px|em|%|in|cm|mm|pc|pt|ex/));
};
Unit.prototype.isEmpty = function () {
    return this.numerator.length === 0 && this.denominator.length === 0;
};
Unit.prototype.isSingular = function() {
    return this.numerator.length <= 1 && this.denominator.length === 0;
};
Unit.prototype.map = function(callback) {
    var i;

    for (i = 0; i < this.numerator.length; i++) {
        this.numerator[i] = callback(this.numerator[i], false);
    }

    for (i = 0; i < this.denominator.length; i++) {
        this.denominator[i] = callback(this.denominator[i], true);
    }
};
Unit.prototype.usedUnits = function() {
    var group, result = {}, mapUnit;

    mapUnit = function (atomicUnit) {
        /*jshint loopfunc:true */
        if (group.hasOwnProperty(atomicUnit) && !result[groupName]) {
            result[groupName] = atomicUnit;
        }

        return atomicUnit;
    };

    for (var groupName in unitConversions) {
        if (unitConversions.hasOwnProperty(groupName)) {
            group = unitConversions[groupName];

            this.map(mapUnit);
        }
    }

    return result;
};
Unit.prototype.cancel = function () {
    var counter = {}, atomicUnit, i, backup;

    for (i = 0; i < this.numerator.length; i++) {
        atomicUnit = this.numerator[i];
        if (!backup) {
            backup = atomicUnit;
        }
        counter[atomicUnit] = (counter[atomicUnit] || 0) + 1;
    }

    for (i = 0; i < this.denominator.length; i++) {
        atomicUnit = this.denominator[i];
        if (!backup) {
            backup = atomicUnit;
        }
        counter[atomicUnit] = (counter[atomicUnit] || 0) - 1;
    }

    this.numerator = [];
    this.denominator = [];

    for (atomicUnit in counter) {
        if (counter.hasOwnProperty(atomicUnit)) {
            var count = counter[atomicUnit];

            if (count > 0) {
                for (i = 0; i < count; i++) {
                    this.numerator.push(atomicUnit);
                }
            } else if (count < 0) {
                for (i = 0; i < -count; i++) {
                    this.denominator.push(atomicUnit);
                }
            }
        }
    }

    if (this.numerator.length === 0 && this.denominator.length === 0 && backup) {
        this.backupUnit = backup;
    }

    this.numerator.sort();
    this.denominator.sort();
};
module.exports = Unit;

},{"../data/unit-conversions.js":5,"./node.js":51}],61:[function(require,module,exports){
var Node = require("./node.js");

var URL = function (val, currentFileInfo, isEvald) {
    this.value = val;
    this.currentFileInfo = currentFileInfo;
    this.isEvald = isEvald;
};
URL.prototype = new Node();
URL.prototype.type = "Url";
URL.prototype.accept = function (visitor) {
    this.value = visitor.visit(this.value);
};
URL.prototype.genCSS = function (env, output) {
    output.add("url(");
    this.value.genCSS(env, output);
    output.add(")");
};
URL.prototype.eval = function (ctx) {
    var val = this.value.eval(ctx),
        rootpath;

    if (!this.isEvald) {
        // Add the base path if the URL is relative
        rootpath = this.currentFileInfo && this.currentFileInfo.rootpath;
        if (rootpath && typeof val.value === "string" && ctx.isPathRelative(val.value)) {
            if (!val.quote) {
                rootpath = rootpath.replace(/[\(\)'"\s]/g, function(match) { return "\\"+match; });
            }
            val.value = rootpath + val.value;
        }

        val.value = ctx.normalizePath(val.value);

        // Add url args if enabled
        if (ctx.urlArgs) {
            if (!val.value.match(/^\s*data:/)) {
                var delimiter = val.value.indexOf('?') === -1 ? '?' : '&';
                var urlArgs = delimiter + ctx.urlArgs;
                if (val.value.indexOf('#') !== -1) {
                    val.value = val.value.replace('#', urlArgs + '#');
                } else {
                    val.value += urlArgs;
                }
            }
        }
    }

    return new(URL)(val, this.currentFileInfo, true);
};
module.exports = URL;

},{"./node.js":51}],62:[function(require,module,exports){
var Node = require("./node.js");

var Value = function (value) {
    this.value = value;
};
Value.prototype = new Node();
Value.prototype.type = "Value";
Value.prototype.accept = function (visitor) {
    if (this.value) {
        this.value = visitor.visitArray(this.value);
    }
};
Value.prototype.eval = function (env) {
    if (this.value.length === 1) {
        return this.value[0].eval(env);
    } else {
        return new(Value)(this.value.map(function (v) {
            return v.eval(env);
        }));
    }
};
Value.prototype.genCSS = function (env, output) {
    var i;
    for(i = 0; i < this.value.length; i++) {
        this.value[i].genCSS(env, output);
        if (i+1 < this.value.length) {
            output.add((env && env.compress) ? ',' : ', ');
        }
    }
};
module.exports = Value;

},{"./node.js":51}],63:[function(require,module,exports){
var Node = require("./node.js");

var Variable = function (name, index, currentFileInfo) {
    this.name = name;
    this.index = index;
    this.currentFileInfo = currentFileInfo || {};
};
Variable.prototype = new Node();
Variable.prototype.type = "Variable";
Variable.prototype.eval = function (env) {
    var variable, name = this.name;

    if (name.indexOf('@@') === 0) {
        name = '@' + new(Variable)(name.slice(1)).eval(env).value;
    }

    if (this.evaluating) {
        throw { type: 'Name',
                message: "Recursive variable definition for " + name,
                filename: this.currentFileInfo.file,
                index: this.index };
    }

    this.evaluating = true;

    variable = this.find(env.frames, function (frame) {
        var v = frame.variable(name);
        if (v) {
            return v.value.eval(env);
        }
    });
    if (variable) {
        this.evaluating = false;
        return variable;
    } else {
        throw { type: 'Name',
                message: "variable " + name + " is undefined",
                filename: this.currentFileInfo.filename,
                index: this.index };
    }
};
Variable.prototype.find = function (obj, fun) {
    for (var i = 0, r; i < obj.length; i++) {
        r = fun.call(obj, obj[i]);
        if (r) { return r; }
    }
    return null;
};
module.exports = Variable;

},{"./node.js":51}],64:[function(require,module,exports){
var tree = require("../tree/index.js"),
    Visitor = require("./visitor.js");

/*jshint loopfunc:true */

var ExtendFinderVisitor = function() {
    this._visitor = new Visitor(this);
    this.contexts = [];
    this.allExtendsStack = [[]];
};

ExtendFinderVisitor.prototype = {
    run: function (root) {
        root = this._visitor.visit(root);
        root.allExtends = this.allExtendsStack[0];
        return root;
    },
    visitRule: function (ruleNode, visitArgs) {
        visitArgs.visitDeeper = false;
    },
    visitMixinDefinition: function (mixinDefinitionNode, visitArgs) {
        visitArgs.visitDeeper = false;
    },
    visitRuleset: function (rulesetNode, visitArgs) {
        if (rulesetNode.root) {
            return;
        }

        var i, j, extend, allSelectorsExtendList = [], extendList;

        // get &:extend(.a); rules which apply to all selectors in this ruleset
        var rules = rulesetNode.rules, ruleCnt = rules ? rules.length : 0;
        for(i = 0; i < ruleCnt; i++) {
            if (rulesetNode.rules[i] instanceof tree.Extend) {
                allSelectorsExtendList.push(rules[i]);
                rulesetNode.extendOnEveryPath = true;
            }
        }

        // now find every selector and apply the extends that apply to all extends
        // and the ones which apply to an individual extend
        var paths = rulesetNode.paths;
        for(i = 0; i < paths.length; i++) {
            var selectorPath = paths[i],
                selector = selectorPath[selectorPath.length - 1],
                selExtendList = selector.extendList;

            extendList = selExtendList ? selExtendList.slice(0).concat(allSelectorsExtendList)
                                       : allSelectorsExtendList;

            if (extendList) {
                extendList = extendList.map(function(allSelectorsExtend) {
                    return allSelectorsExtend.clone();
                });
            }

            for(j = 0; j < extendList.length; j++) {
                this.foundExtends = true;
                extend = extendList[j];
                extend.findSelfSelectors(selectorPath);
                extend.ruleset = rulesetNode;
                if (j === 0) { extend.firstExtendOnThisSelectorPath = true; }
                this.allExtendsStack[this.allExtendsStack.length-1].push(extend);
            }
        }

        this.contexts.push(rulesetNode.selectors);
    },
    visitRulesetOut: function (rulesetNode) {
        if (!rulesetNode.root) {
            this.contexts.length = this.contexts.length - 1;
        }
    },
    visitMedia: function (mediaNode, visitArgs) {
        mediaNode.allExtends = [];
        this.allExtendsStack.push(mediaNode.allExtends);
    },
    visitMediaOut: function (mediaNode) {
        this.allExtendsStack.length = this.allExtendsStack.length - 1;
    },
    visitDirective: function (directiveNode, visitArgs) {
        directiveNode.allExtends = [];
        this.allExtendsStack.push(directiveNode.allExtends);
    },
    visitDirectiveOut: function (directiveNode) {
        this.allExtendsStack.length = this.allExtendsStack.length - 1;
    }
};

var ProcessExtendsVisitor = function() {
    this._visitor = new Visitor(this);
};

ProcessExtendsVisitor.prototype = {
    run: function(root) {
        var extendFinder = new ExtendFinderVisitor();
        extendFinder.run(root);
        if (!extendFinder.foundExtends) { return root; }
        root.allExtends = root.allExtends.concat(this.doExtendChaining(root.allExtends, root.allExtends));
        this.allExtendsStack = [root.allExtends];
        return this._visitor.visit(root);
    },
    doExtendChaining: function (extendsList, extendsListTarget, iterationCount) {
        //
        // chaining is different from normal extension.. if we extend an extend then we are not just copying, altering and pasting
        // the selector we would do normally, but we are also adding an extend with the same target selector
        // this means this new extend can then go and alter other extends
        //
        // this method deals with all the chaining work - without it, extend is flat and doesn't work on other extend selectors
        // this is also the most expensive.. and a match on one selector can cause an extension of a selector we had already processed if
        // we look at each selector at a time, as is done in visitRuleset

        var extendIndex, targetExtendIndex, matches, extendsToAdd = [], newSelector, extendVisitor = this, selectorPath, extend, targetExtend, newExtend;

        iterationCount = iterationCount || 0;

        //loop through comparing every extend with every target extend.
        // a target extend is the one on the ruleset we are looking at copy/edit/pasting in place
        // e.g.  .a:extend(.b) {}  and .b:extend(.c) {} then the first extend extends the second one
        // and the second is the target.
        // the seperation into two lists allows us to process a subset of chains with a bigger set, as is the
        // case when processing media queries
        for(extendIndex = 0; extendIndex < extendsList.length; extendIndex++){
            for(targetExtendIndex = 0; targetExtendIndex < extendsListTarget.length; targetExtendIndex++){

                extend = extendsList[extendIndex];
                targetExtend = extendsListTarget[targetExtendIndex];

                // look for circular references
                if( extend.parent_ids.indexOf( targetExtend.object_id ) >= 0 ){ continue; }

                // find a match in the target extends self selector (the bit before :extend)
                selectorPath = [targetExtend.selfSelectors[0]];
                matches = extendVisitor.findMatch(extend, selectorPath);

                if (matches.length) {

                    // we found a match, so for each self selector..
                    extend.selfSelectors.forEach(function(selfSelector) {

                        // process the extend as usual
                        newSelector = extendVisitor.extendSelector(matches, selectorPath, selfSelector);

                        // but now we create a new extend from it
                        newExtend = new(tree.Extend)(targetExtend.selector, targetExtend.option, 0);
                        newExtend.selfSelectors = newSelector;

                        // add the extend onto the list of extends for that selector
                        newSelector[newSelector.length-1].extendList = [newExtend];

                        // record that we need to add it.
                        extendsToAdd.push(newExtend);
                        newExtend.ruleset = targetExtend.ruleset;

                        //remember its parents for circular references
                        newExtend.parent_ids = newExtend.parent_ids.concat(targetExtend.parent_ids, extend.parent_ids);

                        // only process the selector once.. if we have :extend(.a,.b) then multiple
                        // extends will look at the same selector path, so when extending
                        // we know that any others will be duplicates in terms of what is added to the css
                        if (targetExtend.firstExtendOnThisSelectorPath) {
                            newExtend.firstExtendOnThisSelectorPath = true;
                            targetExtend.ruleset.paths.push(newSelector);
                        }
                    });
                }
            }
        }

        if (extendsToAdd.length) {
            // try to detect circular references to stop a stack overflow.
            // may no longer be needed.
            this.extendChainCount++;
            if (iterationCount > 100) {
                var selectorOne = "{unable to calculate}";
                var selectorTwo = "{unable to calculate}";
                try
                {
                    selectorOne = extendsToAdd[0].selfSelectors[0].toCSS();
                    selectorTwo = extendsToAdd[0].selector.toCSS();
                }
                catch(e) {}
                throw {message: "extend circular reference detected. One of the circular extends is currently:"+selectorOne+":extend(" + selectorTwo+")"};
            }

            // now process the new extends on the existing rules so that we can handle a extending b extending c ectending d extending e...
            return extendsToAdd.concat(extendVisitor.doExtendChaining(extendsToAdd, extendsListTarget, iterationCount+1));
        } else {
            return extendsToAdd;
        }
    },
    visitRule: function (ruleNode, visitArgs) {
        visitArgs.visitDeeper = false;
    },
    visitMixinDefinition: function (mixinDefinitionNode, visitArgs) {
        visitArgs.visitDeeper = false;
    },
    visitSelector: function (selectorNode, visitArgs) {
        visitArgs.visitDeeper = false;
    },
    visitRuleset: function (rulesetNode, visitArgs) {
        if (rulesetNode.root) {
            return;
        }
        var matches, pathIndex, extendIndex, allExtends = this.allExtendsStack[this.allExtendsStack.length-1], selectorsToAdd = [], extendVisitor = this, selectorPath;

        // look at each selector path in the ruleset, find any extend matches and then copy, find and replace

        for(extendIndex = 0; extendIndex < allExtends.length; extendIndex++) {
            for(pathIndex = 0; pathIndex < rulesetNode.paths.length; pathIndex++) {
                selectorPath = rulesetNode.paths[pathIndex];

                // extending extends happens initially, before the main pass
                if (rulesetNode.extendOnEveryPath) { continue; }
                var extendList = selectorPath[selectorPath.length-1].extendList;
                if (extendList && extendList.length) { continue; }

                matches = this.findMatch(allExtends[extendIndex], selectorPath);

                if (matches.length) {

                    allExtends[extendIndex].selfSelectors.forEach(function(selfSelector) {
                        selectorsToAdd.push(extendVisitor.extendSelector(matches, selectorPath, selfSelector));
                    });
                }
            }
        }
        rulesetNode.paths = rulesetNode.paths.concat(selectorsToAdd);
    },
    findMatch: function (extend, haystackSelectorPath) {
        //
        // look through the haystack selector path to try and find the needle - extend.selector
        // returns an array of selector matches that can then be replaced
        //
        var haystackSelectorIndex, hackstackSelector, hackstackElementIndex, haystackElement,
            targetCombinator, i,
            extendVisitor = this,
            needleElements = extend.selector.elements,
            potentialMatches = [], potentialMatch, matches = [];

        // loop through the haystack elements
        for(haystackSelectorIndex = 0; haystackSelectorIndex < haystackSelectorPath.length; haystackSelectorIndex++) {
            hackstackSelector = haystackSelectorPath[haystackSelectorIndex];

            for(hackstackElementIndex = 0; hackstackElementIndex < hackstackSelector.elements.length; hackstackElementIndex++) {

                haystackElement = hackstackSelector.elements[hackstackElementIndex];

                // if we allow elements before our match we can add a potential match every time. otherwise only at the first element.
                if (extend.allowBefore || (haystackSelectorIndex === 0 && hackstackElementIndex === 0)) {
                    potentialMatches.push({pathIndex: haystackSelectorIndex, index: hackstackElementIndex, matched: 0, initialCombinator: haystackElement.combinator});
                }

                for(i = 0; i < potentialMatches.length; i++) {
                    potentialMatch = potentialMatches[i];

                    // selectors add " " onto the first element. When we use & it joins the selectors together, but if we don't
                    // then each selector in haystackSelectorPath has a space before it added in the toCSS phase. so we need to work out
                    // what the resulting combinator will be
                    targetCombinator = haystackElement.combinator.value;
                    if (targetCombinator === '' && hackstackElementIndex === 0) {
                        targetCombinator = ' ';
                    }

                    // if we don't match, null our match to indicate failure
                    if (!extendVisitor.isElementValuesEqual(needleElements[potentialMatch.matched].value, haystackElement.value) ||
                        (potentialMatch.matched > 0 && needleElements[potentialMatch.matched].combinator.value !== targetCombinator)) {
                        potentialMatch = null;
                    } else {
                        potentialMatch.matched++;
                    }

                    // if we are still valid and have finished, test whether we have elements after and whether these are allowed
                    if (potentialMatch) {
                        potentialMatch.finished = potentialMatch.matched === needleElements.length;
                        if (potentialMatch.finished &&
                            (!extend.allowAfter && (hackstackElementIndex+1 < hackstackSelector.elements.length || haystackSelectorIndex+1 < haystackSelectorPath.length))) {
                            potentialMatch = null;
                        }
                    }
                    // if null we remove, if not, we are still valid, so either push as a valid match or continue
                    if (potentialMatch) {
                        if (potentialMatch.finished) {
                            potentialMatch.length = needleElements.length;
                            potentialMatch.endPathIndex = haystackSelectorIndex;
                            potentialMatch.endPathElementIndex = hackstackElementIndex + 1; // index after end of match
                            potentialMatches.length = 0; // we don't allow matches to overlap, so start matching again
                            matches.push(potentialMatch);
                        }
                    } else {
                        potentialMatches.splice(i, 1);
                        i--;
                    }
                }
            }
        }
        return matches;
    },
    isElementValuesEqual: function(elementValue1, elementValue2) {
        if (typeof elementValue1 === "string" || typeof elementValue2 === "string") {
            return elementValue1 === elementValue2;
        }
        if (elementValue1 instanceof tree.Attribute) {
            if (elementValue1.op !== elementValue2.op || elementValue1.key !== elementValue2.key) {
                return false;
            }
            if (!elementValue1.value || !elementValue2.value) {
                if (elementValue1.value || elementValue2.value) {
                    return false;
                }
                return true;
            }
            elementValue1 = elementValue1.value.value || elementValue1.value;
            elementValue2 = elementValue2.value.value || elementValue2.value;
            return elementValue1 === elementValue2;
        }
        elementValue1 = elementValue1.value;
        elementValue2 = elementValue2.value;
        if (elementValue1 instanceof tree.Selector) {
            if (!(elementValue2 instanceof tree.Selector) || elementValue1.elements.length !== elementValue2.elements.length) {
                return false;
            }
            for(var i = 0; i <elementValue1.elements.length; i++) {
                if (elementValue1.elements[i].combinator.value !== elementValue2.elements[i].combinator.value) {
                    if (i !== 0 || (elementValue1.elements[i].combinator.value || ' ') !== (elementValue2.elements[i].combinator.value || ' ')) {
                        return false;
                    }
                }
                if (!this.isElementValuesEqual(elementValue1.elements[i].value, elementValue2.elements[i].value)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    },
    extendSelector:function (matches, selectorPath, replacementSelector) {

        //for a set of matches, replace each match with the replacement selector

        var currentSelectorPathIndex = 0,
            currentSelectorPathElementIndex = 0,
            path = [],
            matchIndex,
            selector,
            firstElement,
            match,
            newElements;

        for (matchIndex = 0; matchIndex < matches.length; matchIndex++) {
            match = matches[matchIndex];
            selector = selectorPath[match.pathIndex];
            firstElement = new tree.Element(
                match.initialCombinator,
                replacementSelector.elements[0].value,
                replacementSelector.elements[0].index,
                replacementSelector.elements[0].currentFileInfo
            );

            if (match.pathIndex > currentSelectorPathIndex && currentSelectorPathElementIndex > 0) {
                path[path.length - 1].elements = path[path.length - 1].elements.concat(selectorPath[currentSelectorPathIndex].elements.slice(currentSelectorPathElementIndex));
                currentSelectorPathElementIndex = 0;
                currentSelectorPathIndex++;
            }

            newElements = selector.elements
                .slice(currentSelectorPathElementIndex, match.index)
                .concat([firstElement])
                .concat(replacementSelector.elements.slice(1));

            if (currentSelectorPathIndex === match.pathIndex && matchIndex > 0) {
                path[path.length - 1].elements =
                    path[path.length - 1].elements.concat(newElements);
            } else {
                path = path.concat(selectorPath.slice(currentSelectorPathIndex, match.pathIndex));

                path.push(new tree.Selector(
                    newElements
                ));
            }
            currentSelectorPathIndex = match.endPathIndex;
            currentSelectorPathElementIndex = match.endPathElementIndex;
            if (currentSelectorPathElementIndex >= selectorPath[currentSelectorPathIndex].elements.length) {
                currentSelectorPathElementIndex = 0;
                currentSelectorPathIndex++;
            }
        }

        if (currentSelectorPathIndex < selectorPath.length && currentSelectorPathElementIndex > 0) {
            path[path.length - 1].elements = path[path.length - 1].elements.concat(selectorPath[currentSelectorPathIndex].elements.slice(currentSelectorPathElementIndex));
            currentSelectorPathIndex++;
        }

        path = path.concat(selectorPath.slice(currentSelectorPathIndex, selectorPath.length));

        return path;
    },
    visitRulesetOut: function (rulesetNode) {
    },
    visitMedia: function (mediaNode, visitArgs) {
        var newAllExtends = mediaNode.allExtends.concat(this.allExtendsStack[this.allExtendsStack.length-1]);
        newAllExtends = newAllExtends.concat(this.doExtendChaining(newAllExtends, mediaNode.allExtends));
        this.allExtendsStack.push(newAllExtends);
    },
    visitMediaOut: function (mediaNode) {
        this.allExtendsStack.length = this.allExtendsStack.length - 1;
    },
    visitDirective: function (directiveNode, visitArgs) {
        var newAllExtends = directiveNode.allExtends.concat(this.allExtendsStack[this.allExtendsStack.length-1]);
        newAllExtends = newAllExtends.concat(this.doExtendChaining(newAllExtends, directiveNode.allExtends));
        this.allExtendsStack.push(newAllExtends);
    },
    visitDirectiveOut: function (directiveNode) {
        this.allExtendsStack.length = this.allExtendsStack.length - 1;
    }
};

module.exports = ProcessExtendsVisitor;

},{"../tree/index.js":43,"./visitor.js":69}],65:[function(require,module,exports){
var contexts = require("../contexts.js"),
    Visitor = require("./visitor.js");

var ImportVisitor = function(importer, finish, evalEnv, onceFileDetectionMap, recursionDetector) {
    this._visitor = new Visitor(this);
    this._importer = importer;
    this._finish = finish;
    this.env = evalEnv || new contexts.evalEnv();
    this.importCount = 0;
    this.onceFileDetectionMap = onceFileDetectionMap || {};
    this.recursionDetector = {};
    if (recursionDetector) {
        for(var fullFilename in recursionDetector) {
            if (recursionDetector.hasOwnProperty(fullFilename)) {
                this.recursionDetector[fullFilename] = true;
            }
        }
    }
};

ImportVisitor.prototype = {
    isReplacing: true,
    run: function (root) {
        var error;
        try {
            // process the contents
            this._visitor.visit(root);
        }
        catch(e) {
            error = e;
        }

        this.isFinished = true;

        if (this.importCount === 0) {
            this._finish(error);
        }
    },
    visitImport: function (importNode, visitArgs) {
        var importVisitor = this,
            evaldImportNode,
            inlineCSS = importNode.options.inline;

        if (!importNode.css || inlineCSS) {

            try {
                evaldImportNode = importNode.evalForImport(this.env);
            } catch(e){
                if (!e.filename) { e.index = importNode.index; e.filename = importNode.currentFileInfo.filename; }
                // attempt to eval properly and treat as css
                importNode.css = true;
                // if that fails, this error will be thrown
                importNode.error = e;
            }

            if (evaldImportNode && (!evaldImportNode.css || inlineCSS)) {
                importNode = evaldImportNode;
                this.importCount++;
                var env = new contexts.evalEnv(this.env, this.env.frames.slice(0));

                if (importNode.options.multiple) {
                    env.importMultiple = true;
                }

                this._importer.push(importNode.getPath(), importNode.currentFileInfo, importNode.options, function (e, root, importedAtRoot, fullPath) {
                    if (e && !e.filename) {
                        e.index = importNode.index; e.filename = importNode.currentFileInfo.filename;
                    }

                    var duplicateImport = importedAtRoot || fullPath in importVisitor.recursionDetector;
                    if (!env.importMultiple) {
                        if (duplicateImport) {
                            importNode.skip = true;
                        } else {
                            importNode.skip = function() {
                                if (fullPath in importVisitor.onceFileDetectionMap) {
                                    return true;
                                }
                                importVisitor.onceFileDetectionMap[fullPath] = true;
                                return false;
                            };
                        }
                    }

                    var subFinish = function(e) {
                        importVisitor.importCount--;

                        if (importVisitor.importCount === 0 && importVisitor.isFinished) {
                            importVisitor._finish(e);
                        }
                    };

                    if (root) {
                        importNode.root = root;
                        importNode.importedFilename = fullPath;

                        if (!inlineCSS && (env.importMultiple || !duplicateImport)) {
                            importVisitor.recursionDetector[fullPath] = true;
                            new(ImportVisitor)(importVisitor._importer, subFinish, env, importVisitor.onceFileDetectionMap, importVisitor.recursionDetector)
                                .run(root);
                            return;
                        }
                    }

                    subFinish();
                });
            }
        }
        visitArgs.visitDeeper = false;
        return importNode;
    },
    visitRule: function (ruleNode, visitArgs) {
        visitArgs.visitDeeper = false;
        return ruleNode;
    },
    visitDirective: function (directiveNode, visitArgs) {
        this.env.frames.unshift(directiveNode);
        return directiveNode;
    },
    visitDirectiveOut: function (directiveNode) {
        this.env.frames.shift();
    },
    visitMixinDefinition: function (mixinDefinitionNode, visitArgs) {
        this.env.frames.unshift(mixinDefinitionNode);
        return mixinDefinitionNode;
    },
    visitMixinDefinitionOut: function (mixinDefinitionNode) {
        this.env.frames.shift();
    },
    visitRuleset: function (rulesetNode, visitArgs) {
        this.env.frames.unshift(rulesetNode);
        return rulesetNode;
    },
    visitRulesetOut: function (rulesetNode) {
        this.env.frames.shift();
    },
    visitMedia: function (mediaNode, visitArgs) {
        this.env.frames.unshift(mediaNode.rules[0]);
        return mediaNode;
    },
    visitMediaOut: function (mediaNode) {
        this.env.frames.shift();
    }
};
module.exports = ImportVisitor;

},{"../contexts.js":2,"./visitor.js":69}],66:[function(require,module,exports){
var visitors = {
    Visitor: require("./visitor"),
    ImportVisitor: require('./import-visitor.js'),
    ExtendVisitor: require('./extend-visitor.js'),
    JoinSelectorVisitor: require('./join-selector-visitor.js'),
    ToCSSVisitor: require('./to-css-visitor.js')
};

module.exports = visitors;

},{"./extend-visitor.js":64,"./import-visitor.js":65,"./join-selector-visitor.js":67,"./to-css-visitor.js":68,"./visitor":69}],67:[function(require,module,exports){
var Visitor = require("./visitor.js");

var JoinSelectorVisitor = function() {
    this.contexts = [[]];
    this._visitor = new Visitor(this);
};

JoinSelectorVisitor.prototype = {
    run: function (root) {
        return this._visitor.visit(root);
    },
    visitRule: function (ruleNode, visitArgs) {
        visitArgs.visitDeeper = false;
    },
    visitMixinDefinition: function (mixinDefinitionNode, visitArgs) {
        visitArgs.visitDeeper = false;
    },

    visitRuleset: function (rulesetNode, visitArgs) {
        var context = this.contexts[this.contexts.length - 1],
            paths = [], selectors;

        this.contexts.push(paths);

        if (! rulesetNode.root) {
            selectors = rulesetNode.selectors;
            if (selectors) {
                selectors = selectors.filter(function(selector) { return selector.getIsOutput(); });
                rulesetNode.selectors = selectors.length ? selectors : (selectors = null);
                if (selectors) { rulesetNode.joinSelectors(paths, context, selectors); }
            }
            if (!selectors) { rulesetNode.rules = null; }
            rulesetNode.paths = paths;
        }
    },
    visitRulesetOut: function (rulesetNode) {
        this.contexts.length = this.contexts.length - 1;
    },
    visitMedia: function (mediaNode, visitArgs) {
        var context = this.contexts[this.contexts.length - 1];
        mediaNode.rules[0].root = (context.length === 0 || context[0].multiMedia);
    }
};

module.exports = JoinSelectorVisitor;

},{"./visitor.js":69}],68:[function(require,module,exports){
var tree = require("../tree/index.js"),
    Visitor = require("./visitor.js");

var ToCSSVisitor = function(env) {
    this._visitor = new Visitor(this);
    this._env = env;
};

ToCSSVisitor.prototype = {
    isReplacing: true,
    run: function (root) {
        return this._visitor.visit(root);
    },

    visitRule: function (ruleNode, visitArgs) {
        if (ruleNode.variable) {
            return [];
        }
        return ruleNode;
    },

    visitMixinDefinition: function (mixinNode, visitArgs) {
        // mixin definitions do not get eval'd - this means they keep state
        // so we have to clear that state here so it isn't used if toCSS is called twice
        mixinNode.frames = [];
        return [];
    },

    visitExtend: function (extendNode, visitArgs) {
        return [];
    },

    visitComment: function (commentNode, visitArgs) {
        if (commentNode.isSilent(this._env)) {
            return [];
        }
        return commentNode;
    },

    visitMedia: function(mediaNode, visitArgs) {
        mediaNode.accept(this._visitor);
        visitArgs.visitDeeper = false;

        if (!mediaNode.rules.length) {
            return [];
        }
        return mediaNode;
    },

    visitDirective: function(directiveNode, visitArgs) {
        if (directiveNode.currentFileInfo.reference && !directiveNode.isReferenced) {
            return [];
        }
        if (directiveNode.name === "@charset") {
            // Only output the debug info together with subsequent @charset definitions
            // a comment (or @media statement) before the actual @charset directive would
            // be considered illegal css as it has to be on the first line
            if (this.charset) {
                if (directiveNode.debugInfo) {
                    var comment = new tree.Comment("/* " + directiveNode.toCSS(this._env).replace(/\n/g, "")+" */\n");
                    comment.debugInfo = directiveNode.debugInfo;
                    return this._visitor.visit(comment);
                }
                return [];
            }
            this.charset = true;
        }
        if (directiveNode.rules && directiveNode.rules.rules) {
            this._mergeRules(directiveNode.rules.rules);
        }
        return directiveNode;
    },

    checkPropertiesInRoot: function(rules) {
        var ruleNode;
        for(var i = 0; i < rules.length; i++) {
            ruleNode = rules[i];
            if (ruleNode instanceof tree.Rule && !ruleNode.variable) {
                throw { message: "properties must be inside selector blocks, they cannot be in the root.",
                    index: ruleNode.index, filename: ruleNode.currentFileInfo ? ruleNode.currentFileInfo.filename : null};
            }
        }
    },

    visitRuleset: function (rulesetNode, visitArgs) {
        var rule, rulesets = [];
        if (rulesetNode.firstRoot) {
            this.checkPropertiesInRoot(rulesetNode.rules);
        }
        if (! rulesetNode.root) {
            if (rulesetNode.paths) {
                rulesetNode.paths = rulesetNode.paths
                    .filter(function(p) {
                        var i;
                        if (p[0].elements[0].combinator.value === ' ') {
                            p[0].elements[0].combinator = new(tree.Combinator)('');
                        }
                        for(i = 0; i < p.length; i++) {
                            if (p[i].getIsReferenced() && p[i].getIsOutput()) {
                                return true;
                            }
                        }
                        return false;
                    });
            }

            // Compile rules and rulesets
            var nodeRules = rulesetNode.rules, nodeRuleCnt = nodeRules ? nodeRules.length : 0;
            for (var i = 0; i < nodeRuleCnt; ) {
                rule = nodeRules[i];
                if (rule && rule.rules) {
                    // visit because we are moving them out from being a child
                    rulesets.push(this._visitor.visit(rule));
                    nodeRules.splice(i, 1);
                    nodeRuleCnt--;
                    continue;
                }
                i++;
            }
            // accept the visitor to remove rules and refactor itself
            // then we can decide now whether we want it or not
            if (nodeRuleCnt > 0) {
                rulesetNode.accept(this._visitor);
            } else {
                rulesetNode.rules = null;
            }
            visitArgs.visitDeeper = false;

            nodeRules = rulesetNode.rules;
            if (nodeRules) {
                this._mergeRules(nodeRules);
                nodeRules = rulesetNode.rules;
            }
            if (nodeRules) {
                this._removeDuplicateRules(nodeRules);
                nodeRules = rulesetNode.rules;
            }

            // now decide whether we keep the ruleset
            if (nodeRules && nodeRules.length > 0 && rulesetNode.paths.length > 0) {
                rulesets.splice(0, 0, rulesetNode);
            }
        } else {
            rulesetNode.accept(this._visitor);
            visitArgs.visitDeeper = false;
            if (rulesetNode.firstRoot || (rulesetNode.rules && rulesetNode.rules.length > 0)) {
                rulesets.splice(0, 0, rulesetNode);
            }
        }
        if (rulesets.length === 1) {
            return rulesets[0];
        }
        return rulesets;
    },

    _removeDuplicateRules: function(rules) {
        if (!rules) { return; }

        // remove duplicates
        var ruleCache = {},
            ruleList, rule, i;

        for(i = rules.length - 1; i >= 0 ; i--) {
            rule = rules[i];
            if (rule instanceof tree.Rule) {
                if (!ruleCache[rule.name]) {
                    ruleCache[rule.name] = rule;
                } else {
                    ruleList = ruleCache[rule.name];
                    if (ruleList instanceof tree.Rule) {
                        ruleList = ruleCache[rule.name] = [ruleCache[rule.name].toCSS(this._env)];
                    }
                    var ruleCSS = rule.toCSS(this._env);
                    if (ruleList.indexOf(ruleCSS) !== -1) {
                        rules.splice(i, 1);
                    } else {
                        ruleList.push(ruleCSS);
                    }
                }
            }
        }
    },

    _mergeRules: function (rules) {
        if (!rules) { return; }

        var groups = {},
            parts,
            rule,
            key;

        for (var i = 0; i < rules.length; i++) {
            rule = rules[i];

            if ((rule instanceof tree.Rule) && rule.merge) {
                key = [rule.name,
                    rule.important ? "!" : ""].join(",");

                if (!groups[key]) {
                    groups[key] = [];
                } else {
                    rules.splice(i--, 1);
                }

                groups[key].push(rule);
            }
        }

        Object.keys(groups).map(function (k) {

            function toExpression(values) {
                return new (tree.Expression)(values.map(function (p) {
                    return p.value;
                }));
            }

            function toValue(values) {
                return new (tree.Value)(values.map(function (p) {
                    return p;
                }));
            }

            parts = groups[k];

            if (parts.length > 1) {
                rule = parts[0];
                var spacedGroups = [];
                var lastSpacedGroup = [];
                parts.map(function (p) {
                if (p.merge==="+") {
                    if (lastSpacedGroup.length > 0) {
                            spacedGroups.push(toExpression(lastSpacedGroup));
                        }
                        lastSpacedGroup = [];
                    }
                    lastSpacedGroup.push(p);
                });
                spacedGroups.push(toExpression(lastSpacedGroup));
                rule.value = toValue(spacedGroups);
            }
        });
    }
};

module.exports = ToCSSVisitor;

},{"../tree/index.js":43,"./visitor.js":69}],69:[function(require,module,exports){
var tree = require("../tree/index.js");

var _visitArgs = { visitDeeper: true },
    _hasIndexed = false;

function _noop(node) {
    return node;
}

function indexNodeTypes(parent, ticker) {
    // add .typeIndex to tree node types for lookup table
    var key, child;
    for (key in parent) {
        if (parent.hasOwnProperty(key)) {
            child = parent[key];
            switch (typeof child) {
                case "function":
                    // ignore bound functions directly on tree which do not have a prototype
                    // or aren't nodes
                    if (child.prototype && child.prototype.type) {
                        child.prototype.typeIndex = ticker++;
                    }
                    break;
                case "object":
                    ticker = indexNodeTypes(child, ticker);
                    break;
            }
        }
    }
    return ticker;
}

var Visitor = function(implementation) {
    this._implementation = implementation;
    this._visitFnCache = [];

    if (!_hasIndexed) {
        indexNodeTypes(tree, 1);
        _hasIndexed = true;
    }
};

Visitor.prototype = {
    visit: function(node) {
        if (!node) {
            return node;
        }

        var nodeTypeIndex = node.typeIndex;
        if (!nodeTypeIndex) {
            return node;
        }

        var visitFnCache = this._visitFnCache,
            impl = this._implementation,
            aryIndx = nodeTypeIndex << 1,
            outAryIndex = aryIndx | 1,
            func = visitFnCache[aryIndx],
            funcOut = visitFnCache[outAryIndex],
            visitArgs = _visitArgs,
            fnName;

        visitArgs.visitDeeper = true;

        if (!func) {
            fnName = "visit" + node.type;
            func = impl[fnName] || _noop;
            funcOut = impl[fnName + "Out"] || _noop;
            visitFnCache[aryIndx] = func;
            visitFnCache[outAryIndex] = funcOut;
        }

        if (func !== _noop) {
            var newNode = func.call(impl, node, visitArgs);
            if (impl.isReplacing) {
                node = newNode;
            }
        }

        if (visitArgs.visitDeeper && node && node.accept) {
            node.accept(this);
        }

        if (funcOut != _noop) {
            funcOut.call(impl, node);
        }

        return node;
    },
    visitArray: function(nodes, nonReplacing) {
        if (!nodes) {
            return nodes;
        }

        var cnt = nodes.length, i;

        // Non-replacing
        if (nonReplacing || !this._implementation.isReplacing) {
            for (i = 0; i < cnt; i++) {
                this.visit(nodes[i]);
            }
            return nodes;
        }

        // Replacing
        var out = [];
        for (i = 0; i < cnt; i++) {
            var evald = this.visit(nodes[i]);
            if (!evald.splice) {
                out.push(evald);
            } else if (evald.length) {
                this.flatten(evald, out);
            }
        }
        return out;
    },
    flatten: function(arr, out) {
        if (!out) {
            out = [];
        }

        var cnt, i, item,
            nestedCnt, j, nestedItem;

        for (i = 0, cnt = arr.length; i < cnt; i++) {
            item = arr[i];
            if (!item.splice) {
                out.push(item);
                continue;
            }

            for (j = 0, nestedCnt = item.length; j < nestedCnt; j++) {
                nestedItem = item[j];
                if (!nestedItem.splice) {
                    out.push(nestedItem);
                } else if (nestedItem.length) {
                    this.flatten(nestedItem, out);
                }
            }
        }

        return out;
    }
};
module.exports = Visitor;

},{"../tree/index.js":43}]},{},[1])
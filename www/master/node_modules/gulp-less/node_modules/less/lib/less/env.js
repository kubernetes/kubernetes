(function (tree) {

    var parseCopyProperties = [
        'paths',            // option - unmodified - paths to search for imports on
        'optimization',     // option - optimization level (for the chunker)
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
        'javascriptEnabled',// option - whether JavaScript is enabled. if undefined, defaults to true
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

    tree.parseEnv = function(options) {
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
        'urlArgs'         // whether to add args into url tokens
        ];

    tree.evalEnv = function(options, frames) {
        copyFromOriginal(options, this, evalCopyProperties);

        this.frames = frames || [];
    };

    tree.evalEnv.prototype.inParenthesis = function () {
        if (!this.parensStack) {
            this.parensStack = [];
        }
        this.parensStack.push(true);
    };

    tree.evalEnv.prototype.outOfParenthesis = function () {
        this.parensStack.pop();
    };

    tree.evalEnv.prototype.isMathOn = function () {
        return this.strictMath ? (this.parensStack && this.parensStack.length) : true;
    };

    tree.evalEnv.prototype.isPathRelative = function (path) {
        return !/^(?:[a-z-]+:|\/)/.test(path);
    };

    tree.evalEnv.prototype.normalizePath = function( path ) {
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

    var copyFromOriginal = function(original, destination, propertiesToCopy) {
        if (!original) { return; }

        for(var i = 0; i < propertiesToCopy.length; i++) {
            if (original.hasOwnProperty(propertiesToCopy[i])) {
                destination[propertiesToCopy[i]] = original[propertiesToCopy[i]];
            }
        }
    };

})(require('./tree'));

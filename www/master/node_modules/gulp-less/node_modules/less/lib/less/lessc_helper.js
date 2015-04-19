// lessc_helper.js
//
//      helper functions for lessc
var lessc_helper = {

    //Stylize a string
    stylize : function(str, style) {
        var styles = {
            'reset'     : [0,   0],
            'bold'      : [1,  22],
            'inverse'   : [7,  27],
            'underline' : [4,  24],
            'yellow'    : [33, 39],
            'green'     : [32, 39],
            'red'       : [31, 39],
            'grey'      : [90, 39]
        };
        return '\033[' + styles[style][0] + 'm' + str +
               '\033[' + styles[style][1] + 'm';
    },

    //Print command line options
    printUsage: function() {
        console.log("usage: lessc [option option=parameter ...] <source> [destination]");
        console.log("");
        console.log("If source is set to `-' (dash or hyphen-minus), input is read from stdin.");
        console.log("");
        console.log("options:");
        console.log("  -h, --help               Print help (this message) and exit.");
        console.log("  --include-path=PATHS     Set include paths. Separated by `:'. Use `;' on Windows.");
        console.log("  -M, --depends            Output a makefile import dependency list to stdout");
        console.log("  --no-color               Disable colorized output.");
        console.log("  --no-ie-compat           Disable IE compatibility checks.");
        console.log("  --no-js                  Disable JavaScript in less files");
        console.log("  -l, --lint               Syntax check only (lint).");
        console.log("  -s, --silent             Suppress output of error messages.");
        console.log("  --strict-imports         Force evaluation of imports.");
        console.log("  --insecure               Allow imports from insecure https hosts.");
        console.log("  -v, --version            Print version number and exit.");
        console.log("  -x, --compress           Compress output by removing some whitespaces.");
        console.log("  --clean-css              Compress output using clean-css");
        console.log("  --clean-option=opt:val   Pass an option to clean css, using CLI arguments from ");
        console.log("                           https://github.com/GoalSmashers/clean-css e.g.");
        console.log("                           --clean-option=--selectors-merge-mode:ie8");
        console.log("                           and to switch on advanced use --clean-option=--advanced");
        console.log("  --source-map[=FILENAME]  Outputs a v3 sourcemap to the filename (or output filename.map)");
        console.log("  --source-map-rootpath=X  adds this path onto the sourcemap filename and less file paths");
        console.log("  --source-map-basepath=X  Sets sourcemap base path, defaults to current working directory.");
        console.log("  --source-map-less-inline puts the less files into the map instead of referencing them");
        console.log("  --source-map-map-inline  puts the map (and any less files) into the output css file");
        console.log("  --source-map-url=URL     the complete url and filename put in the less file");
        console.log("  -rp, --rootpath=URL      Set rootpath for url rewriting in relative imports and urls.");
        console.log("                           Works with or without the relative-urls option.");
        console.log("  -ru, --relative-urls     re-write relative urls to the base less file.");
        console.log("  -sm=on|off               Turn on or off strict math, where in strict mode, math");
        console.log("  --strict-math=on|off     requires brackets. This option may default to on and then");
        console.log("                           be removed in the future.");
        console.log("  -su=on|off               Allow mixed units, e.g. 1px+1em or 1px*1px which have units");
        console.log("  --strict-units=on|off    that cannot be represented.");
        console.log("  --global-var='VAR=VALUE' Defines a variable that can be referenced by the file.");
        console.log("  --modify-var='VAR=VALUE' Modifies a variable already declared in the file.");
        console.log("  --url-args='QUERYSTRING' Adds params into url tokens (e.g. 42, cb=42 or 'a=1&b=2')");
        console.log("");
        console.log("-------------------------- Deprecated ----------------");
        console.log("  -O0, -O1, -O2            Set the parser's optimization level. The lower");
        console.log("                           the number, the less nodes it will create in the");
        console.log("                           tree. This could matter for debugging, or if you");
        console.log("                           want to access the individual nodes in the tree.");
        console.log("  --line-numbers=TYPE      Outputs filename and line numbers.");
        console.log("                           TYPE can be either 'comments', which will output");
        console.log("                           the debug info within comments, 'mediaquery'");
        console.log("                           that will output the information within a fake");
        console.log("                           media query which is compatible with the SASS");
        console.log("                           format, and 'all' which will do both.");
        console.log("  --verbose                Be verbose.");
        console.log("");
        console.log("Report bugs to: http://github.com/less/less.js/issues");
        console.log("Home page: <http://lesscss.org/>");
    }
};

// Exports helper functions
for (var h in lessc_helper) { if (lessc_helper.hasOwnProperty(h)) { exports[h] = lessc_helper[h]; }}

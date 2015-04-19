/* Silly noise to be able to load in source code without doing an XMLHttpReq */var jquery111 = "/*!\n\
 * jQuery JavaScript Library v1.11.1\n\
 * http://jquery.com/\n\
 *\n\
 * Includes Sizzle.js\n\
 * http://sizzlejs.com/\n\
 *\n\
 * Copyright 2005, 2014 jQuery Foundation, Inc. and other contributors\n\
 * Released under the MIT license\n\
 * http://jquery.org/license\n\
 *\n\
 * Date: 2014-05-01T17:42Z\n\
 */\n\
\n\
(function( global, factory ) {\n\
\n\
\tif ( typeof module === \"object\" && typeof module.exports === \"object\" ) {\n\
\t\t// For CommonJS and CommonJS-like environments where a proper window is present,\n\
\t\t// execute the factory and get jQuery\n\
\t\t// For environments that do not inherently posses a window with a document\n\
\t\t// (such as Node.js), expose a jQuery-making factory as module.exports\n\
\t\t// This accentuates the need for the creation of a real window\n\
\t\t// e.g. var jQuery = require(\"jquery\")(window);\n\
\t\t// See ticket #14549 for more info\n\
\t\tmodule.exports = global.document ?\n\
\t\t\tfactory( global, true ) :\n\
\t\t\tfunction( w ) {\n\
\t\t\t\tif ( !w.document ) {\n\
\t\t\t\t\tthrow new Error( \"jQuery requires a window with a document\" );\n\
\t\t\t\t}\n\
\t\t\t\treturn factory( w );\n\
\t\t\t};\n\
\t} else {\n\
\t\tfactory( global );\n\
\t}\n\
\n\
// Pass this if window is not defined yet\n\
}(typeof window !== \"undefined\" ? window : this, function( window, noGlobal ) {\n\
\n\
// Can't do this because several apps including ASP.NET trace\n\
// the stack via arguments.caller.callee and Firefox dies if\n\
// you try to trace through \"use strict\" call chains. (#13335)\n\
// Support: Firefox 18+\n\
//\n\
\n\
var deletedIds = [];\n\
\n\
var slice = deletedIds.slice;\n\
\n\
var concat = deletedIds.concat;\n\
\n\
var push = deletedIds.push;\n\
\n\
var indexOf = deletedIds.indexOf;\n\
\n\
var class2type = {};\n\
\n\
var toString = class2type.toString;\n\
\n\
var hasOwn = class2type.hasOwnProperty;\n\
\n\
var support = {};\n\
\n\
\n\
\n\
var\n\
\tversion = \"1.11.1\",\n\
\n\
\t// Define a local copy of jQuery\n\
\tjQuery = function( selector, context ) {\n\
\t\t// The jQuery object is actually just the init constructor 'enhanced'\n\
\t\t// Need init if jQuery is called (just allow error to be thrown if not included)\n\
\t\treturn new jQuery.fn.init( selector, context );\n\
\t},\n\
\n\
\t// Support: Android<4.1, IE<9\n\
\t// Make sure we trim BOM and NBSP\n\
\trtrim = /^[\\s\\uFEFF\\xA0]+|[\\s\\uFEFF\\xA0]+$/g,\n\
\n\
\t// Matches dashed string for camelizing\n\
\trmsPrefix = /^-ms-/,\n\
\trdashAlpha = /-([\\da-z])/gi,\n\
\n\
\t// Used by jQuery.camelCase as callback to replace()\n\
\tfcamelCase = function( all, letter ) {\n\
\t\treturn letter.toUpperCase();\n\
\t};\n\
\n\
jQuery.fn = jQuery.prototype = {\n\
\t// The current version of jQuery being used\n\
\tjquery: version,\n\
\n\
\tconstructor: jQuery,\n\
\n\
\t// Start with an empty selector\n\
\tselector: \"\",\n\
\n\
\t// The default length of a jQuery object is 0\n\
\tlength: 0,\n\
\n\
\ttoArray: function() {\n\
\t\treturn slice.call( this );\n\
\t},\n\
\n\
\t// Get the Nth element in the matched element set OR\n\
\t// Get the whole matched element set as a clean array\n\
\tget: function( num ) {\n\
\t\treturn num != null ?\n\
\n\
\t\t\t// Return just the one element from the set\n\
\t\t\t( num < 0 ? this[ num + this.length ] : this[ num ] ) :\n\
\n\
\t\t\t// Return all the elements in a clean array\n\
\t\t\tslice.call( this );\n\
\t},\n\
\n\
\t// Take an array of elements and push it onto the stack\n\
\t// (returning the new matched element set)\n\
\tpushStack: function( elems ) {\n\
\n\
\t\t// Build a new jQuery matched element set\n\
\t\tvar ret = jQuery.merge( this.constructor(), elems );\n\
\n\
\t\t// Add the old object onto the stack (as a reference)\n\
\t\tret.prevObject = this;\n\
\t\tret.context = this.context;\n\
\n\
\t\t// Return the newly-formed element set\n\
\t\treturn ret;\n\
\t},\n\
\n\
\t// Execute a callback for every element in the matched set.\n\
\t// (You can seed the arguments with an array of args, but this is\n\
\t// only used internally.)\n\
\teach: function( callback, args ) {\n\
\t\treturn jQuery.each( this, callback, args );\n\
\t},\n\
\n\
\tmap: function( callback ) {\n\
\t\treturn this.pushStack( jQuery.map(this, function( elem, i ) {\n\
\t\t\treturn callback.call( elem, i, elem );\n\
\t\t}));\n\
\t},\n\
\n\
\tslice: function() {\n\
\t\treturn this.pushStack( slice.apply( this, arguments ) );\n\
\t},\n\
\n\
\tfirst: function() {\n\
\t\treturn this.eq( 0 );\n\
\t},\n\
\n\
\tlast: function() {\n\
\t\treturn this.eq( -1 );\n\
\t},\n\
\n\
\teq: function( i ) {\n\
\t\tvar len = this.length,\n\
\t\t\tj = +i + ( i < 0 ? len : 0 );\n\
\t\treturn this.pushStack( j >= 0 && j < len ? [ this[j] ] : [] );\n\
\t},\n\
\n\
\tend: function() {\n\
\t\treturn this.prevObject || this.constructor(null);\n\
\t},\n\
\n\
\t// For internal use only.\n\
\t// Behaves like an Array's method, not like a jQuery method.\n\
\tpush: push,\n\
\tsort: deletedIds.sort,\n\
\tsplice: deletedIds.splice\n\
};\n\
\n\
jQuery.extend = jQuery.fn.extend = function() {\n\
\tvar src, copyIsArray, copy, name, options, clone,\n\
\t\ttarget = arguments[0] || {},\n\
\t\ti = 1,\n\
\t\tlength = arguments.length,\n\
\t\tdeep = false;\n\
\n\
\t// Handle a deep copy situation\n\
\tif ( typeof target === \"boolean\" ) {\n\
\t\tdeep = target;\n\
\n\
\t\t// skip the boolean and the target\n\
\t\ttarget = arguments[ i ] || {};\n\
\t\ti++;\n\
\t}\n\
\n\
\t// Handle case when target is a string or something (possible in deep copy)\n\
\tif ( typeof target !== \"object\" && !jQuery.isFunction(target) ) {\n\
\t\ttarget = {};\n\
\t}\n\
\n\
\t// extend jQuery itself if only one argument is passed\n\
\tif ( i === length ) {\n\
\t\ttarget = this;\n\
\t\ti--;\n\
\t}\n\
\n\
\tfor ( ; i < length; i++ ) {\n\
\t\t// Only deal with non-null/undefined values\n\
\t\tif ( (options = arguments[ i ]) != null ) {\n\
\t\t\t// Extend the base object\n\
\t\t\tfor ( name in options ) {\n\
\t\t\t\tsrc = target[ name ];\n\
\t\t\t\tcopy = options[ name ];\n\
\n\
\t\t\t\t// Prevent never-ending loop\n\
\t\t\t\tif ( target === copy ) {\n\
\t\t\t\t\tcontinue;\n\
\t\t\t\t}\n\
\n\
\t\t\t\t// Recurse if we're merging plain objects or arrays\n\
\t\t\t\tif ( deep && copy && ( jQuery.isPlainObject(copy) || (copyIsArray = jQuery.isArray(copy)) ) ) {\n\
\t\t\t\t\tif ( copyIsArray ) {\n\
\t\t\t\t\t\tcopyIsArray = false;\n\
\t\t\t\t\t\tclone = src && jQuery.isArray(src) ? src : [];\n\
\n\
\t\t\t\t\t} else {\n\
\t\t\t\t\t\tclone = src && jQuery.isPlainObject(src) ? src : {};\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// Never move original objects, clone them\n\
\t\t\t\t\ttarget[ name ] = jQuery.extend( deep, clone, copy );\n\
\n\
\t\t\t\t// Don't bring in undefined values\n\
\t\t\t\t} else if ( copy !== undefined ) {\n\
\t\t\t\t\ttarget[ name ] = copy;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\t// Return the modified object\n\
\treturn target;\n\
};\n\
\n\
jQuery.extend({\n\
\t// Unique for each copy of jQuery on the page\n\
\texpando: \"jQuery\" + ( version + Math.random() ).replace( /\\D/g, \"\" ),\n\
\n\
\t// Assume jQuery is ready without the ready module\n\
\tisReady: true,\n\
\n\
\terror: function( msg ) {\n\
\t\tthrow new Error( msg );\n\
\t},\n\
\n\
\tnoop: function() {},\n\
\n\
\t// See test/unit/core.js for details concerning isFunction.\n\
\t// Since version 1.3, DOM methods and functions like alert\n\
\t// aren't supported. They return false on IE (#2968).\n\
\tisFunction: function( obj ) {\n\
\t\treturn jQuery.type(obj) === \"function\";\n\
\t},\n\
\n\
\tisArray: Array.isArray || function( obj ) {\n\
\t\treturn jQuery.type(obj) === \"array\";\n\
\t},\n\
\n\
\tisWindow: function( obj ) {\n\
\t\t/* jshint eqeqeq: false */\n\
\t\treturn obj != null && obj == obj.window;\n\
\t},\n\
\n\
\tisNumeric: function( obj ) {\n\
\t\t// parseFloat NaNs numeric-cast false positives (null|true|false|\"\")\n\
\t\t// ...but misinterprets leading-number strings, particularly hex literals (\"0x...\")\n\
\t\t// subtraction forces infinities to NaN\n\
\t\treturn !jQuery.isArray( obj ) && obj - parseFloat( obj ) >= 0;\n\
\t},\n\
\n\
\tisEmptyObject: function( obj ) {\n\
\t\tvar name;\n\
\t\tfor ( name in obj ) {\n\
\t\t\treturn false;\n\
\t\t}\n\
\t\treturn true;\n\
\t},\n\
\n\
\tisPlainObject: function( obj ) {\n\
\t\tvar key;\n\
\n\
\t\t// Must be an Object.\n\
\t\t// Because of IE, we also have to check the presence of the constructor property.\n\
\t\t// Make sure that DOM nodes and window objects don't pass through, as well\n\
\t\tif ( !obj || jQuery.type(obj) !== \"object\" || obj.nodeType || jQuery.isWindow( obj ) ) {\n\
\t\t\treturn false;\n\
\t\t}\n\
\n\
\t\ttry {\n\
\t\t\t// Not own constructor property must be Object\n\
\t\t\tif ( obj.constructor &&\n\
\t\t\t\t!hasOwn.call(obj, \"constructor\") &&\n\
\t\t\t\t!hasOwn.call(obj.constructor.prototype, \"isPrototypeOf\") ) {\n\
\t\t\t\treturn false;\n\
\t\t\t}\n\
\t\t} catch ( e ) {\n\
\t\t\t// IE8,9 Will throw exceptions on certain host objects #9897\n\
\t\t\treturn false;\n\
\t\t}\n\
\n\
\t\t// Support: IE<9\n\
\t\t// Handle iteration over inherited properties before own properties.\n\
\t\tif ( support.ownLast ) {\n\
\t\t\tfor ( key in obj ) {\n\
\t\t\t\treturn hasOwn.call( obj, key );\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Own properties are enumerated firstly, so to speed up,\n\
\t\t// if last one is own, then all properties are own.\n\
\t\tfor ( key in obj ) {}\n\
\n\
\t\treturn key === undefined || hasOwn.call( obj, key );\n\
\t},\n\
\n\
\ttype: function( obj ) {\n\
\t\tif ( obj == null ) {\n\
\t\t\treturn obj + \"\";\n\
\t\t}\n\
\t\treturn typeof obj === \"object\" || typeof obj === \"function\" ?\n\
\t\t\tclass2type[ toString.call(obj) ] || \"object\" :\n\
\t\t\ttypeof obj;\n\
\t},\n\
\n\
\t// Evaluates a script in a global context\n\
\t// Workarounds based on findings by Jim Driscoll\n\
\t// http://weblogs.java.net/blog/driscoll/archive/2009/09/08/eval-javascript-global-context\n\
\tglobalEval: function( data ) {\n\
\t\tif ( data && jQuery.trim( data ) ) {\n\
\t\t\t// We use execScript on Internet Explorer\n\
\t\t\t// We use an anonymous function so that context is window\n\
\t\t\t// rather than jQuery in Firefox\n\
\t\t\t( window.execScript || function( data ) {\n\
\t\t\t\twindow[ \"eval\" ].call( window, data );\n\
\t\t\t} )( data );\n\
\t\t}\n\
\t},\n\
\n\
\t// Convert dashed to camelCase; used by the css and data modules\n\
\t// Microsoft forgot to hump their vendor prefix (#9572)\n\
\tcamelCase: function( string ) {\n\
\t\treturn string.replace( rmsPrefix, \"ms-\" ).replace( rdashAlpha, fcamelCase );\n\
\t},\n\
\n\
\tnodeName: function( elem, name ) {\n\
\t\treturn elem.nodeName && elem.nodeName.toLowerCase() === name.toLowerCase();\n\
\t},\n\
\n\
\t// args is for internal usage only\n\
\teach: function( obj, callback, args ) {\n\
\t\tvar value,\n\
\t\t\ti = 0,\n\
\t\t\tlength = obj.length,\n\
\t\t\tisArray = isArraylike( obj );\n\
\n\
\t\tif ( args ) {\n\
\t\t\tif ( isArray ) {\n\
\t\t\t\tfor ( ; i < length; i++ ) {\n\
\t\t\t\t\tvalue = callback.apply( obj[ i ], args );\n\
\n\
\t\t\t\t\tif ( value === false ) {\n\
\t\t\t\t\t\tbreak;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t} else {\n\
\t\t\t\tfor ( i in obj ) {\n\
\t\t\t\t\tvalue = callback.apply( obj[ i ], args );\n\
\n\
\t\t\t\t\tif ( value === false ) {\n\
\t\t\t\t\t\tbreak;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t// A special, fast, case for the most common use of each\n\
\t\t} else {\n\
\t\t\tif ( isArray ) {\n\
\t\t\t\tfor ( ; i < length; i++ ) {\n\
\t\t\t\t\tvalue = callback.call( obj[ i ], i, obj[ i ] );\n\
\n\
\t\t\t\t\tif ( value === false ) {\n\
\t\t\t\t\t\tbreak;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t} else {\n\
\t\t\t\tfor ( i in obj ) {\n\
\t\t\t\t\tvalue = callback.call( obj[ i ], i, obj[ i ] );\n\
\n\
\t\t\t\t\tif ( value === false ) {\n\
\t\t\t\t\t\tbreak;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn obj;\n\
\t},\n\
\n\
\t// Support: Android<4.1, IE<9\n\
\ttrim: function( text ) {\n\
\t\treturn text == null ?\n\
\t\t\t\"\" :\n\
\t\t\t( text + \"\" ).replace( rtrim, \"\" );\n\
\t},\n\
\n\
\t// results is for internal usage only\n\
\tmakeArray: function( arr, results ) {\n\
\t\tvar ret = results || [];\n\
\n\
\t\tif ( arr != null ) {\n\
\t\t\tif ( isArraylike( Object(arr) ) ) {\n\
\t\t\t\tjQuery.merge( ret,\n\
\t\t\t\t\ttypeof arr === \"string\" ?\n\
\t\t\t\t\t[ arr ] : arr\n\
\t\t\t\t);\n\
\t\t\t} else {\n\
\t\t\t\tpush.call( ret, arr );\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn ret;\n\
\t},\n\
\n\
\tinArray: function( elem, arr, i ) {\n\
\t\tvar len;\n\
\n\
\t\tif ( arr ) {\n\
\t\t\tif ( indexOf ) {\n\
\t\t\t\treturn indexOf.call( arr, elem, i );\n\
\t\t\t}\n\
\n\
\t\t\tlen = arr.length;\n\
\t\t\ti = i ? i < 0 ? Math.max( 0, len + i ) : i : 0;\n\
\n\
\t\t\tfor ( ; i < len; i++ ) {\n\
\t\t\t\t// Skip accessing in sparse arrays\n\
\t\t\t\tif ( i in arr && arr[ i ] === elem ) {\n\
\t\t\t\t\treturn i;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn -1;\n\
\t},\n\
\n\
\tmerge: function( first, second ) {\n\
\t\tvar len = +second.length,\n\
\t\t\tj = 0,\n\
\t\t\ti = first.length;\n\
\n\
\t\twhile ( j < len ) {\n\
\t\t\tfirst[ i++ ] = second[ j++ ];\n\
\t\t}\n\
\n\
\t\t// Support: IE<9\n\
\t\t// Workaround casting of .length to NaN on otherwise arraylike objects (e.g., NodeLists)\n\
\t\tif ( len !== len ) {\n\
\t\t\twhile ( second[j] !== undefined ) {\n\
\t\t\t\tfirst[ i++ ] = second[ j++ ];\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\tfirst.length = i;\n\
\n\
\t\treturn first;\n\
\t},\n\
\n\
\tgrep: function( elems, callback, invert ) {\n\
\t\tvar callbackInverse,\n\
\t\t\tmatches = [],\n\
\t\t\ti = 0,\n\
\t\t\tlength = elems.length,\n\
\t\t\tcallbackExpect = !invert;\n\
\n\
\t\t// Go through the array, only saving the items\n\
\t\t// that pass the validator function\n\
\t\tfor ( ; i < length; i++ ) {\n\
\t\t\tcallbackInverse = !callback( elems[ i ], i );\n\
\t\t\tif ( callbackInverse !== callbackExpect ) {\n\
\t\t\t\tmatches.push( elems[ i ] );\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn matches;\n\
\t},\n\
\n\
\t// arg is for internal usage only\n\
\tmap: function( elems, callback, arg ) {\n\
\t\tvar value,\n\
\t\t\ti = 0,\n\
\t\t\tlength = elems.length,\n\
\t\t\tisArray = isArraylike( elems ),\n\
\t\t\tret = [];\n\
\n\
\t\t// Go through the array, translating each of the items to their new values\n\
\t\tif ( isArray ) {\n\
\t\t\tfor ( ; i < length; i++ ) {\n\
\t\t\t\tvalue = callback( elems[ i ], i, arg );\n\
\n\
\t\t\t\tif ( value != null ) {\n\
\t\t\t\t\tret.push( value );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t// Go through every key on the object,\n\
\t\t} else {\n\
\t\t\tfor ( i in elems ) {\n\
\t\t\t\tvalue = callback( elems[ i ], i, arg );\n\
\n\
\t\t\t\tif ( value != null ) {\n\
\t\t\t\t\tret.push( value );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Flatten any nested arrays\n\
\t\treturn concat.apply( [], ret );\n\
\t},\n\
\n\
\t// A global GUID counter for objects\n\
\tguid: 1,\n\
\n\
\t// Bind a function to a context, optionally partially applying any\n\
\t// arguments.\n\
\tproxy: function( fn, context ) {\n\
\t\tvar args, proxy, tmp;\n\
\n\
\t\tif ( typeof context === \"string\" ) {\n\
\t\t\ttmp = fn[ context ];\n\
\t\t\tcontext = fn;\n\
\t\t\tfn = tmp;\n\
\t\t}\n\
\n\
\t\t// Quick check to determine if target is callable, in the spec\n\
\t\t// this throws a TypeError, but we will just return undefined.\n\
\t\tif ( !jQuery.isFunction( fn ) ) {\n\
\t\t\treturn undefined;\n\
\t\t}\n\
\n\
\t\t// Simulated bind\n\
\t\targs = slice.call( arguments, 2 );\n\
\t\tproxy = function() {\n\
\t\t\treturn fn.apply( context || this, args.concat( slice.call( arguments ) ) );\n\
\t\t};\n\
\n\
\t\t// Set the guid of unique handler to the same of original handler, so it can be removed\n\
\t\tproxy.guid = fn.guid = fn.guid || jQuery.guid++;\n\
\n\
\t\treturn proxy;\n\
\t},\n\
\n\
\tnow: function() {\n\
\t\treturn +( new Date() );\n\
\t},\n\
\n\
\t// jQuery.support is not used in Core but other projects attach their\n\
\t// properties to it so it needs to exist.\n\
\tsupport: support\n\
});\n\
\n\
// Populate the class2type map\n\
jQuery.each(\"Boolean Number String Function Array Date RegExp Object Error\".split(\" \"), function(i, name) {\n\
\tclass2type[ \"[object \" + name + \"]\" ] = name.toLowerCase();\n\
});\n\
\n\
function isArraylike( obj ) {\n\
\tvar length = obj.length,\n\
\t\ttype = jQuery.type( obj );\n\
\n\
\tif ( type === \"function\" || jQuery.isWindow( obj ) ) {\n\
\t\treturn false;\n\
\t}\n\
\n\
\tif ( obj.nodeType === 1 && length ) {\n\
\t\treturn true;\n\
\t}\n\
\n\
\treturn type === \"array\" || length === 0 ||\n\
\t\ttypeof length === \"number\" && length > 0 && ( length - 1 ) in obj;\n\
}\n\
var Sizzle =\n\
/*!\n\
 * Sizzle CSS Selector Engine v1.10.19\n\
 * http://sizzlejs.com/\n\
 *\n\
 * Copyright 2013 jQuery Foundation, Inc. and other contributors\n\
 * Released under the MIT license\n\
 * http://jquery.org/license\n\
 *\n\
 * Date: 2014-04-18\n\
 */\n\
(function( window ) {\n\
\n\
var i,\n\
\tsupport,\n\
\tExpr,\n\
\tgetText,\n\
\tisXML,\n\
\ttokenize,\n\
\tcompile,\n\
\tselect,\n\
\toutermostContext,\n\
\tsortInput,\n\
\thasDuplicate,\n\
\n\
\t// Local document vars\n\
\tsetDocument,\n\
\tdocument,\n\
\tdocElem,\n\
\tdocumentIsHTML,\n\
\trbuggyQSA,\n\
\trbuggyMatches,\n\
\tmatches,\n\
\tcontains,\n\
\n\
\t// Instance-specific data\n\
\texpando = \"sizzle\" + -(new Date()),\n\
\tpreferredDoc = window.document,\n\
\tdirruns = 0,\n\
\tdone = 0,\n\
\tclassCache = createCache(),\n\
\ttokenCache = createCache(),\n\
\tcompilerCache = createCache(),\n\
\tsortOrder = function( a, b ) {\n\
\t\tif ( a === b ) {\n\
\t\t\thasDuplicate = true;\n\
\t\t}\n\
\t\treturn 0;\n\
\t},\n\
\n\
\t// General-purpose constants\n\
\tstrundefined = typeof undefined,\n\
\tMAX_NEGATIVE = 1 << 31,\n\
\n\
\t// Instance methods\n\
\thasOwn = ({}).hasOwnProperty,\n\
\tarr = [],\n\
\tpop = arr.pop,\n\
\tpush_native = arr.push,\n\
\tpush = arr.push,\n\
\tslice = arr.slice,\n\
\t// Use a stripped-down indexOf if we can't use a native one\n\
\tindexOf = arr.indexOf || function( elem ) {\n\
\t\tvar i = 0,\n\
\t\t\tlen = this.length;\n\
\t\tfor ( ; i < len; i++ ) {\n\
\t\t\tif ( this[i] === elem ) {\n\
\t\t\t\treturn i;\n\
\t\t\t}\n\
\t\t}\n\
\t\treturn -1;\n\
\t},\n\
\n\
\tbooleans = \"checked|selected|async|autofocus|autoplay|controls|defer|disabled|hidden|ismap|loop|multiple|open|readonly|required|scoped\",\n\
\n\
\t// Regular expressions\n\
\n\
\t// Whitespace characters http://www.w3.org/TR/css3-selectors/#whitespace\n\
\twhitespace = \"[\\\\x20\\\\t\\\\r\\\\n\
\\\\f]\",\n\
\t// http://www.w3.org/TR/css3-syntax/#characters\n\
\tcharacterEncoding = \"(?:\\\\\\\\.|[\\\\w-]|[^\\\\x00-\\\\xa0])+\",\n\
\n\
\t// Loosely modeled on CSS identifier characters\n\
\t// An unquoted value should be a CSS identifier http://www.w3.org/TR/css3-selectors/#attribute-selectors\n\
\t// Proper syntax: http://www.w3.org/TR/CSS21/syndata.html#value-def-identifier\n\
\tidentifier = characterEncoding.replace( \"w\", \"w#\" ),\n\
\n\
\t// Attribute selectors: http://www.w3.org/TR/selectors/#attribute-selectors\n\
\tattributes = \"\\\\[\" + whitespace + \"*(\" + characterEncoding + \")(?:\" + whitespace +\n\
\t\t// Operator (capture 2)\n\
\t\t\"*([*^$|!~]?=)\" + whitespace +\n\
\t\t// \"Attribute values must be CSS identifiers [capture 5] or strings [capture 3 or capture 4]\"\n\
\t\t\"*(?:'((?:\\\\\\\\.|[^\\\\\\\\'])*)'|\\\"((?:\\\\\\\\.|[^\\\\\\\\\\\"])*)\\\"|(\" + identifier + \"))|)\" + whitespace +\n\
\t\t\"*\\\\]\",\n\
\n\
\tpseudos = \":(\" + characterEncoding + \")(?:\\\\((\" +\n\
\t\t// To reduce the number of selectors needing tokenize in the preFilter, prefer arguments:\n\
\t\t// 1. quoted (capture 3; capture 4 or capture 5)\n\
\t\t\"('((?:\\\\\\\\.|[^\\\\\\\\'])*)'|\\\"((?:\\\\\\\\.|[^\\\\\\\\\\\"])*)\\\")|\" +\n\
\t\t// 2. simple (capture 6)\n\
\t\t\"((?:\\\\\\\\.|[^\\\\\\\\()[\\\\]]|\" + attributes + \")*)|\" +\n\
\t\t// 3. anything else (capture 2)\n\
\t\t\".*\" +\n\
\t\t\")\\\\)|)\",\n\
\n\
\t// Leading and non-escaped trailing whitespace, capturing some non-whitespace characters preceding the latter\n\
\trtrim = new RegExp( \"^\" + whitespace + \"+|((?:^|[^\\\\\\\\])(?:\\\\\\\\.)*)\" + whitespace + \"+$\", \"g\" ),\n\
\n\
\trcomma = new RegExp( \"^\" + whitespace + \"*,\" + whitespace + \"*\" ),\n\
\trcombinators = new RegExp( \"^\" + whitespace + \"*([>+~]|\" + whitespace + \")\" + whitespace + \"*\" ),\n\
\n\
\trattributeQuotes = new RegExp( \"=\" + whitespace + \"*([^\\\\]'\\\"]*?)\" + whitespace + \"*\\\\]\", \"g\" ),\n\
\n\
\trpseudo = new RegExp( pseudos ),\n\
\tridentifier = new RegExp( \"^\" + identifier + \"$\" ),\n\
\n\
\tmatchExpr = {\n\
\t\t\"ID\": new RegExp( \"^#(\" + characterEncoding + \")\" ),\n\
\t\t\"CLASS\": new RegExp( \"^\\\\.(\" + characterEncoding + \")\" ),\n\
\t\t\"TAG\": new RegExp( \"^(\" + characterEncoding.replace( \"w\", \"w*\" ) + \")\" ),\n\
\t\t\"ATTR\": new RegExp( \"^\" + attributes ),\n\
\t\t\"PSEUDO\": new RegExp( \"^\" + pseudos ),\n\
\t\t\"CHILD\": new RegExp( \"^:(only|first|last|nth|nth-last)-(child|of-type)(?:\\\\(\" + whitespace +\n\
\t\t\t\"*(even|odd|(([+-]|)(\\\\d*)n|)\" + whitespace + \"*(?:([+-]|)\" + whitespace +\n\
\t\t\t\"*(\\\\d+)|))\" + whitespace + \"*\\\\)|)\", \"i\" ),\n\
\t\t\"bool\": new RegExp( \"^(?:\" + booleans + \")$\", \"i\" ),\n\
\t\t// For use in libraries implementing .is()\n\
\t\t// We use this for POS matching in `select`\n\
\t\t\"needsContext\": new RegExp( \"^\" + whitespace + \"*[>+~]|:(even|odd|eq|gt|lt|nth|first|last)(?:\\\\(\" +\n\
\t\t\twhitespace + \"*((?:-\\\\d)?\\\\d*)\" + whitespace + \"*\\\\)|)(?=[^-]|$)\", \"i\" )\n\
\t},\n\
\n\
\trinputs = /^(?:input|select|textarea|button)$/i,\n\
\trheader = /^h\\d$/i,\n\
\n\
\trnative = /^[^{]+\\{\\s*\\[native \\w/,\n\
\n\
\t// Easily-parseable/retrievable ID or TAG or CLASS selectors\n\
\trquickExpr = /^(?:#([\\w-]+)|(\\w+)|\\.([\\w-]+))$/,\n\
\n\
\trsibling = /[+~]/,\n\
\trescape = /'|\\\\/g,\n\
\n\
\t// CSS escapes http://www.w3.org/TR/CSS21/syndata.html#escaped-characters\n\
\trunescape = new RegExp( \"\\\\\\\\([\\\\da-f]{1,6}\" + whitespace + \"?|(\" + whitespace + \")|.)\", \"ig\" ),\n\
\tfunescape = function( _, escaped, escapedWhitespace ) {\n\
\t\tvar high = \"0x\" + escaped - 0x10000;\n\
\t\t// NaN means non-codepoint\n\
\t\t// Support: Firefox<24\n\
\t\t// Workaround erroneous numeric interpretation of +\"0x\"\n\
\t\treturn high !== high || escapedWhitespace ?\n\
\t\t\tescaped :\n\
\t\t\thigh < 0 ?\n\
\t\t\t\t// BMP codepoint\n\
\t\t\t\tString.fromCharCode( high + 0x10000 ) :\n\
\t\t\t\t// Supplemental Plane codepoint (surrogate pair)\n\
\t\t\t\tString.fromCharCode( high >> 10 | 0xD800, high & 0x3FF | 0xDC00 );\n\
\t};\n\
\n\
// Optimize for push.apply( _, NodeList )\n\
try {\n\
\tpush.apply(\n\
\t\t(arr = slice.call( preferredDoc.childNodes )),\n\
\t\tpreferredDoc.childNodes\n\
\t);\n\
\t// Support: Android<4.0\n\
\t// Detect silently failing push.apply\n\
\tarr[ preferredDoc.childNodes.length ].nodeType;\n\
} catch ( e ) {\n\
\tpush = { apply: arr.length ?\n\
\n\
\t\t// Leverage slice if possible\n\
\t\tfunction( target, els ) {\n\
\t\t\tpush_native.apply( target, slice.call(els) );\n\
\t\t} :\n\
\n\
\t\t// Support: IE<9\n\
\t\t// Otherwise append directly\n\
\t\tfunction( target, els ) {\n\
\t\t\tvar j = target.length,\n\
\t\t\t\ti = 0;\n\
\t\t\t// Can't trust NodeList.length\n\
\t\t\twhile ( (target[j++] = els[i++]) ) {}\n\
\t\t\ttarget.length = j - 1;\n\
\t\t}\n\
\t};\n\
}\n\
\n\
function Sizzle( selector, context, results, seed ) {\n\
\tvar match, elem, m, nodeType,\n\
\t\t// QSA vars\n\
\t\ti, groups, old, nid, newContext, newSelector;\n\
\n\
\tif ( ( context ? context.ownerDocument || context : preferredDoc ) !== document ) {\n\
\t\tsetDocument( context );\n\
\t}\n\
\n\
\tcontext = context || document;\n\
\tresults = results || [];\n\
\n\
\tif ( !selector || typeof selector !== \"string\" ) {\n\
\t\treturn results;\n\
\t}\n\
\n\
\tif ( (nodeType = context.nodeType) !== 1 && nodeType !== 9 ) {\n\
\t\treturn [];\n\
\t}\n\
\n\
\tif ( documentIsHTML && !seed ) {\n\
\n\
\t\t// Shortcuts\n\
\t\tif ( (match = rquickExpr.exec( selector )) ) {\n\
\t\t\t// Speed-up: Sizzle(\"#ID\")\n\
\t\t\tif ( (m = match[1]) ) {\n\
\t\t\t\tif ( nodeType === 9 ) {\n\
\t\t\t\t\telem = context.getElementById( m );\n\
\t\t\t\t\t// Check parentNode to catch when Blackberry 4.6 returns\n\
\t\t\t\t\t// nodes that are no longer in the document (jQuery #6963)\n\
\t\t\t\t\tif ( elem && elem.parentNode ) {\n\
\t\t\t\t\t\t// Handle the case where IE, Opera, and Webkit return items\n\
\t\t\t\t\t\t// by name instead of ID\n\
\t\t\t\t\t\tif ( elem.id === m ) {\n\
\t\t\t\t\t\t\tresults.push( elem );\n\
\t\t\t\t\t\t\treturn results;\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t} else {\n\
\t\t\t\t\t\treturn results;\n\
\t\t\t\t\t}\n\
\t\t\t\t} else {\n\
\t\t\t\t\t// Context is not a document\n\
\t\t\t\t\tif ( context.ownerDocument && (elem = context.ownerDocument.getElementById( m )) &&\n\
\t\t\t\t\t\tcontains( context, elem ) && elem.id === m ) {\n\
\t\t\t\t\t\tresults.push( elem );\n\
\t\t\t\t\t\treturn results;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\n\
\t\t\t// Speed-up: Sizzle(\"TAG\")\n\
\t\t\t} else if ( match[2] ) {\n\
\t\t\t\tpush.apply( results, context.getElementsByTagName( selector ) );\n\
\t\t\t\treturn results;\n\
\n\
\t\t\t// Speed-up: Sizzle(\".CLASS\")\n\
\t\t\t} else if ( (m = match[3]) && support.getElementsByClassName && context.getElementsByClassName ) {\n\
\t\t\t\tpush.apply( results, context.getElementsByClassName( m ) );\n\
\t\t\t\treturn results;\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// QSA path\n\
\t\tif ( support.qsa && (!rbuggyQSA || !rbuggyQSA.test( selector )) ) {\n\
\t\t\tnid = old = expando;\n\
\t\t\tnewContext = context;\n\
\t\t\tnewSelector = nodeType === 9 && selector;\n\
\n\
\t\t\t// qSA works strangely on Element-rooted queries\n\
\t\t\t// We can work around this by specifying an extra ID on the root\n\
\t\t\t// and working up from there (Thanks to Andrew Dupont for the technique)\n\
\t\t\t// IE 8 doesn't work on object elements\n\
\t\t\tif ( nodeType === 1 && context.nodeName.toLowerCase() !== \"object\" ) {\n\
\t\t\t\tgroups = tokenize( selector );\n\
\n\
\t\t\t\tif ( (old = context.getAttribute(\"id\")) ) {\n\
\t\t\t\t\tnid = old.replace( rescape, \"\\\\$&\" );\n\
\t\t\t\t} else {\n\
\t\t\t\t\tcontext.setAttribute( \"id\", nid );\n\
\t\t\t\t}\n\
\t\t\t\tnid = \"[id='\" + nid + \"'] \";\n\
\n\
\t\t\t\ti = groups.length;\n\
\t\t\t\twhile ( i-- ) {\n\
\t\t\t\t\tgroups[i] = nid + toSelector( groups[i] );\n\
\t\t\t\t}\n\
\t\t\t\tnewContext = rsibling.test( selector ) && testContext( context.parentNode ) || context;\n\
\t\t\t\tnewSelector = groups.join(\",\");\n\
\t\t\t}\n\
\n\
\t\t\tif ( newSelector ) {\n\
\t\t\t\ttry {\n\
\t\t\t\t\tpush.apply( results,\n\
\t\t\t\t\t\tnewContext.querySelectorAll( newSelector )\n\
\t\t\t\t\t);\n\
\t\t\t\t\treturn results;\n\
\t\t\t\t} catch(qsaError) {\n\
\t\t\t\t} finally {\n\
\t\t\t\t\tif ( !old ) {\n\
\t\t\t\t\t\tcontext.removeAttribute(\"id\");\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\t// All others\n\
\treturn select( selector.replace( rtrim, \"$1\" ), context, results, seed );\n\
}\n\
\n\
/**\n\
 * Create key-value caches of limited size\n\
 * @returns {Function(string, Object)} Returns the Object data after storing it on itself with\n\
 *\tproperty name the (space-suffixed) string and (if the cache is larger than Expr.cacheLength)\n\
 *\tdeleting the oldest entry\n\
 */\n\
function createCache() {\n\
\tvar keys = [];\n\
\n\
\tfunction cache( key, value ) {\n\
\t\t// Use (key + \" \") to avoid collision with native prototype properties (see Issue #157)\n\
\t\tif ( keys.push( key + \" \" ) > Expr.cacheLength ) {\n\
\t\t\t// Only keep the most recent entries\n\
\t\t\tdelete cache[ keys.shift() ];\n\
\t\t}\n\
\t\treturn (cache[ key + \" \" ] = value);\n\
\t}\n\
\treturn cache;\n\
}\n\
\n\
/**\n\
 * Mark a function for special use by Sizzle\n\
 * @param {Function} fn The function to mark\n\
 */\n\
function markFunction( fn ) {\n\
\tfn[ expando ] = true;\n\
\treturn fn;\n\
}\n\
\n\
/**\n\
 * Support testing using an element\n\
 * @param {Function} fn Passed the created div and expects a boolean result\n\
 */\n\
function assert( fn ) {\n\
\tvar div = document.createElement(\"div\");\n\
\n\
\ttry {\n\
\t\treturn !!fn( div );\n\
\t} catch (e) {\n\
\t\treturn false;\n\
\t} finally {\n\
\t\t// Remove from its parent by default\n\
\t\tif ( div.parentNode ) {\n\
\t\t\tdiv.parentNode.removeChild( div );\n\
\t\t}\n\
\t\t// release memory in IE\n\
\t\tdiv = null;\n\
\t}\n\
}\n\
\n\
/**\n\
 * Adds the same handler for all of the specified attrs\n\
 * @param {String} attrs Pipe-separated list of attributes\n\
 * @param {Function} handler The method that will be applied\n\
 */\n\
function addHandle( attrs, handler ) {\n\
\tvar arr = attrs.split(\"|\"),\n\
\t\ti = attrs.length;\n\
\n\
\twhile ( i-- ) {\n\
\t\tExpr.attrHandle[ arr[i] ] = handler;\n\
\t}\n\
}\n\
\n\
/**\n\
 * Checks document order of two siblings\n\
 * @param {Element} a\n\
 * @param {Element} b\n\
 * @returns {Number} Returns less than 0 if a precedes b, greater than 0 if a follows b\n\
 */\n\
function siblingCheck( a, b ) {\n\
\tvar cur = b && a,\n\
\t\tdiff = cur && a.nodeType === 1 && b.nodeType === 1 &&\n\
\t\t\t( ~b.sourceIndex || MAX_NEGATIVE ) -\n\
\t\t\t( ~a.sourceIndex || MAX_NEGATIVE );\n\
\n\
\t// Use IE sourceIndex if available on both nodes\n\
\tif ( diff ) {\n\
\t\treturn diff;\n\
\t}\n\
\n\
\t// Check if b follows a\n\
\tif ( cur ) {\n\
\t\twhile ( (cur = cur.nextSibling) ) {\n\
\t\t\tif ( cur === b ) {\n\
\t\t\t\treturn -1;\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\treturn a ? 1 : -1;\n\
}\n\
\n\
/**\n\
 * Returns a function to use in pseudos for input types\n\
 * @param {String} type\n\
 */\n\
function createInputPseudo( type ) {\n\
\treturn function( elem ) {\n\
\t\tvar name = elem.nodeName.toLowerCase();\n\
\t\treturn name === \"input\" && elem.type === type;\n\
\t};\n\
}\n\
\n\
/**\n\
 * Returns a function to use in pseudos for buttons\n\
 * @param {String} type\n\
 */\n\
function createButtonPseudo( type ) {\n\
\treturn function( elem ) {\n\
\t\tvar name = elem.nodeName.toLowerCase();\n\
\t\treturn (name === \"input\" || name === \"button\") && elem.type === type;\n\
\t};\n\
}\n\
\n\
/**\n\
 * Returns a function to use in pseudos for positionals\n\
 * @param {Function} fn\n\
 */\n\
function createPositionalPseudo( fn ) {\n\
\treturn markFunction(function( argument ) {\n\
\t\targument = +argument;\n\
\t\treturn markFunction(function( seed, matches ) {\n\
\t\t\tvar j,\n\
\t\t\t\tmatchIndexes = fn( [], seed.length, argument ),\n\
\t\t\t\ti = matchIndexes.length;\n\
\n\
\t\t\t// Match elements found at the specified indexes\n\
\t\t\twhile ( i-- ) {\n\
\t\t\t\tif ( seed[ (j = matchIndexes[i]) ] ) {\n\
\t\t\t\t\tseed[j] = !(matches[j] = seed[j]);\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t});\n\
\t});\n\
}\n\
\n\
/**\n\
 * Checks a node for validity as a Sizzle context\n\
 * @param {Element|Object=} context\n\
 * @returns {Element|Object|Boolean} The input node if acceptable, otherwise a falsy value\n\
 */\n\
function testContext( context ) {\n\
\treturn context && typeof context.getElementsByTagName !== strundefined && context;\n\
}\n\
\n\
// Expose support vars for convenience\n\
support = Sizzle.support = {};\n\
\n\
/**\n\
 * Detects XML nodes\n\
 * @param {Element|Object} elem An element or a document\n\
 * @returns {Boolean} True iff elem is a non-HTML XML node\n\
 */\n\
isXML = Sizzle.isXML = function( elem ) {\n\
\t// documentElement is verified for cases where it doesn't yet exist\n\
\t// (such as loading iframes in IE - #4833)\n\
\tvar documentElement = elem && (elem.ownerDocument || elem).documentElement;\n\
\treturn documentElement ? documentElement.nodeName !== \"HTML\" : false;\n\
};\n\
\n\
/**\n\
 * Sets document-related variables once based on the current document\n\
 * @param {Element|Object} [doc] An element or document object to use to set the document\n\
 * @returns {Object} Returns the current document\n\
 */\n\
setDocument = Sizzle.setDocument = function( node ) {\n\
\tvar hasCompare,\n\
\t\tdoc = node ? node.ownerDocument || node : preferredDoc,\n\
\t\tparent = doc.defaultView;\n\
\n\
\t// If no document and documentElement is available, return\n\
\tif ( doc === document || doc.nodeType !== 9 || !doc.documentElement ) {\n\
\t\treturn document;\n\
\t}\n\
\n\
\t// Set our document\n\
\tdocument = doc;\n\
\tdocElem = doc.documentElement;\n\
\n\
\t// Support tests\n\
\tdocumentIsHTML = !isXML( doc );\n\
\n\
\t// Support: IE>8\n\
\t// If iframe document is assigned to \"document\" variable and if iframe has been reloaded,\n\
\t// IE will throw \"permission denied\" error when accessing \"document\" variable, see jQuery #13936\n\
\t// IE6-8 do not support the defaultView property so parent will be undefined\n\
\tif ( parent && parent !== parent.top ) {\n\
\t\t// IE11 does not have attachEvent, so all must suffer\n\
\t\tif ( parent.addEventListener ) {\n\
\t\t\tparent.addEventListener( \"unload\", function() {\n\
\t\t\t\tsetDocument();\n\
\t\t\t}, false );\n\
\t\t} else if ( parent.attachEvent ) {\n\
\t\t\tparent.attachEvent( \"onunload\", function() {\n\
\t\t\t\tsetDocument();\n\
\t\t\t});\n\
\t\t}\n\
\t}\n\
\n\
\t/* Attributes\n\
\t---------------------------------------------------------------------- */\n\
\n\
\t// Support: IE<8\n\
\t// Verify that getAttribute really returns attributes and not properties (excepting IE8 booleans)\n\
\tsupport.attributes = assert(function( div ) {\n\
\t\tdiv.className = \"i\";\n\
\t\treturn !div.getAttribute(\"className\");\n\
\t});\n\
\n\
\t/* getElement(s)By*\n\
\t---------------------------------------------------------------------- */\n\
\n\
\t// Check if getElementsByTagName(\"*\") returns only elements\n\
\tsupport.getElementsByTagName = assert(function( div ) {\n\
\t\tdiv.appendChild( doc.createComment(\"\") );\n\
\t\treturn !div.getElementsByTagName(\"*\").length;\n\
\t});\n\
\n\
\t// Check if getElementsByClassName can be trusted\n\
\tsupport.getElementsByClassName = rnative.test( doc.getElementsByClassName ) && assert(function( div ) {\n\
\t\tdiv.innerHTML = \"<div class='a'></div><div class='a i'></div>\";\n\
\n\
\t\t// Support: Safari<4\n\
\t\t// Catch class over-caching\n\
\t\tdiv.firstChild.className = \"i\";\n\
\t\t// Support: Opera<10\n\
\t\t// Catch gEBCN failure to find non-leading classes\n\
\t\treturn div.getElementsByClassName(\"i\").length === 2;\n\
\t});\n\
\n\
\t// Support: IE<10\n\
\t// Check if getElementById returns elements by name\n\
\t// The broken getElementById methods don't pick up programatically-set names,\n\
\t// so use a roundabout getElementsByName test\n\
\tsupport.getById = assert(function( div ) {\n\
\t\tdocElem.appendChild( div ).id = expando;\n\
\t\treturn !doc.getElementsByName || !doc.getElementsByName( expando ).length;\n\
\t});\n\
\n\
\t// ID find and filter\n\
\tif ( support.getById ) {\n\
\t\tExpr.find[\"ID\"] = function( id, context ) {\n\
\t\t\tif ( typeof context.getElementById !== strundefined && documentIsHTML ) {\n\
\t\t\t\tvar m = context.getElementById( id );\n\
\t\t\t\t// Check parentNode to catch when Blackberry 4.6 returns\n\
\t\t\t\t// nodes that are no longer in the document #6963\n\
\t\t\t\treturn m && m.parentNode ? [ m ] : [];\n\
\t\t\t}\n\
\t\t};\n\
\t\tExpr.filter[\"ID\"] = function( id ) {\n\
\t\t\tvar attrId = id.replace( runescape, funescape );\n\
\t\t\treturn function( elem ) {\n\
\t\t\t\treturn elem.getAttribute(\"id\") === attrId;\n\
\t\t\t};\n\
\t\t};\n\
\t} else {\n\
\t\t// Support: IE6/7\n\
\t\t// getElementById is not reliable as a find shortcut\n\
\t\tdelete Expr.find[\"ID\"];\n\
\n\
\t\tExpr.filter[\"ID\"] =  function( id ) {\n\
\t\t\tvar attrId = id.replace( runescape, funescape );\n\
\t\t\treturn function( elem ) {\n\
\t\t\t\tvar node = typeof elem.getAttributeNode !== strundefined && elem.getAttributeNode(\"id\");\n\
\t\t\t\treturn node && node.value === attrId;\n\
\t\t\t};\n\
\t\t};\n\
\t}\n\
\n\
\t// Tag\n\
\tExpr.find[\"TAG\"] = support.getElementsByTagName ?\n\
\t\tfunction( tag, context ) {\n\
\t\t\tif ( typeof context.getElementsByTagName !== strundefined ) {\n\
\t\t\t\treturn context.getElementsByTagName( tag );\n\
\t\t\t}\n\
\t\t} :\n\
\t\tfunction( tag, context ) {\n\
\t\t\tvar elem,\n\
\t\t\t\ttmp = [],\n\
\t\t\t\ti = 0,\n\
\t\t\t\tresults = context.getElementsByTagName( tag );\n\
\n\
\t\t\t// Filter out possible comments\n\
\t\t\tif ( tag === \"*\" ) {\n\
\t\t\t\twhile ( (elem = results[i++]) ) {\n\
\t\t\t\t\tif ( elem.nodeType === 1 ) {\n\
\t\t\t\t\t\ttmp.push( elem );\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\n\
\t\t\t\treturn tmp;\n\
\t\t\t}\n\
\t\t\treturn results;\n\
\t\t};\n\
\n\
\t// Class\n\
\tExpr.find[\"CLASS\"] = support.getElementsByClassName && function( className, context ) {\n\
\t\tif ( typeof context.getElementsByClassName !== strundefined && documentIsHTML ) {\n\
\t\t\treturn context.getElementsByClassName( className );\n\
\t\t}\n\
\t};\n\
\n\
\t/* QSA/matchesSelector\n\
\t---------------------------------------------------------------------- */\n\
\n\
\t// QSA and matchesSelector support\n\
\n\
\t// matchesSelector(:active) reports false when true (IE9/Opera 11.5)\n\
\trbuggyMatches = [];\n\
\n\
\t// qSa(:focus) reports false when true (Chrome 21)\n\
\t// We allow this because of a bug in IE8/9 that throws an error\n\
\t// whenever `document.activeElement` is accessed on an iframe\n\
\t// So, we allow :focus to pass through QSA all the time to avoid the IE error\n\
\t// See http://bugs.jquery.com/ticket/13378\n\
\trbuggyQSA = [];\n\
\n\
\tif ( (support.qsa = rnative.test( doc.querySelectorAll )) ) {\n\
\t\t// Build QSA regex\n\
\t\t// Regex strategy adopted from Diego Perini\n\
\t\tassert(function( div ) {\n\
\t\t\t// Select is set to empty string on purpose\n\
\t\t\t// This is to test IE's treatment of not explicitly\n\
\t\t\t// setting a boolean content attribute,\n\
\t\t\t// since its presence should be enough\n\
\t\t\t// http://bugs.jquery.com/ticket/12359\n\
\t\t\tdiv.innerHTML = \"<select msallowclip=''><option selected=''></option></select>\";\n\
\n\
\t\t\t// Support: IE8, Opera 11-12.16\n\
\t\t\t// Nothing should be selected when empty strings follow ^= or $= or *=\n\
\t\t\t// The test attribute must be unknown in Opera but \"safe\" for WinRT\n\
\t\t\t// http://msdn.microsoft.com/en-us/library/ie/hh465388.aspx#attribute_section\n\
\t\t\tif ( div.querySelectorAll(\"[msallowclip^='']\").length ) {\n\
\t\t\t\trbuggyQSA.push( \"[*^$]=\" + whitespace + \"*(?:''|\\\"\\\")\" );\n\
\t\t\t}\n\
\n\
\t\t\t// Support: IE8\n\
\t\t\t// Boolean attributes and \"value\" are not treated correctly\n\
\t\t\tif ( !div.querySelectorAll(\"[selected]\").length ) {\n\
\t\t\t\trbuggyQSA.push( \"\\\\[\" + whitespace + \"*(?:value|\" + booleans + \")\" );\n\
\t\t\t}\n\
\n\
\t\t\t// Webkit/Opera - :checked should return selected option elements\n\
\t\t\t// http://www.w3.org/TR/2011/REC-css3-selectors-20110929/#checked\n\
\t\t\t// IE8 throws error here and will not see later tests\n\
\t\t\tif ( !div.querySelectorAll(\":checked\").length ) {\n\
\t\t\t\trbuggyQSA.push(\":checked\");\n\
\t\t\t}\n\
\t\t});\n\
\n\
\t\tassert(function( div ) {\n\
\t\t\t// Support: Windows 8 Native Apps\n\
\t\t\t// The type and name attributes are restricted during .innerHTML assignment\n\
\t\t\tvar input = doc.createElement(\"input\");\n\
\t\t\tinput.setAttribute( \"type\", \"hidden\" );\n\
\t\t\tdiv.appendChild( input ).setAttribute( \"name\", \"D\" );\n\
\n\
\t\t\t// Support: IE8\n\
\t\t\t// Enforce case-sensitivity of name attribute\n\
\t\t\tif ( div.querySelectorAll(\"[name=d]\").length ) {\n\
\t\t\t\trbuggyQSA.push( \"name\" + whitespace + \"*[*^$|!~]?=\" );\n\
\t\t\t}\n\
\n\
\t\t\t// FF 3.5 - :enabled/:disabled and hidden elements (hidden elements are still enabled)\n\
\t\t\t// IE8 throws error here and will not see later tests\n\
\t\t\tif ( !div.querySelectorAll(\":enabled\").length ) {\n\
\t\t\t\trbuggyQSA.push( \":enabled\", \":disabled\" );\n\
\t\t\t}\n\
\n\
\t\t\t// Opera 10-11 does not throw on post-comma invalid pseudos\n\
\t\t\tdiv.querySelectorAll(\"*,:x\");\n\
\t\t\trbuggyQSA.push(\",.*:\");\n\
\t\t});\n\
\t}\n\
\n\
\tif ( (support.matchesSelector = rnative.test( (matches = docElem.matches ||\n\
\t\tdocElem.webkitMatchesSelector ||\n\
\t\tdocElem.mozMatchesSelector ||\n\
\t\tdocElem.oMatchesSelector ||\n\
\t\tdocElem.msMatchesSelector) )) ) {\n\
\n\
\t\tassert(function( div ) {\n\
\t\t\t// Check to see if it's possible to do matchesSelector\n\
\t\t\t// on a disconnected node (IE 9)\n\
\t\t\tsupport.disconnectedMatch = matches.call( div, \"div\" );\n\
\n\
\t\t\t// This should fail with an exception\n\
\t\t\t// Gecko does not error, returns false instead\n\
\t\t\tmatches.call( div, \"[s!='']:x\" );\n\
\t\t\trbuggyMatches.push( \"!=\", pseudos );\n\
\t\t});\n\
\t}\n\
\n\
\trbuggyQSA = rbuggyQSA.length && new RegExp( rbuggyQSA.join(\"|\") );\n\
\trbuggyMatches = rbuggyMatches.length && new RegExp( rbuggyMatches.join(\"|\") );\n\
\n\
\t/* Contains\n\
\t---------------------------------------------------------------------- */\n\
\thasCompare = rnative.test( docElem.compareDocumentPosition );\n\
\n\
\t// Element contains another\n\
\t// Purposefully does not implement inclusive descendent\n\
\t// As in, an element does not contain itself\n\
\tcontains = hasCompare || rnative.test( docElem.contains ) ?\n\
\t\tfunction( a, b ) {\n\
\t\t\tvar adown = a.nodeType === 9 ? a.documentElement : a,\n\
\t\t\t\tbup = b && b.parentNode;\n\
\t\t\treturn a === bup || !!( bup && bup.nodeType === 1 && (\n\
\t\t\t\tadown.contains ?\n\
\t\t\t\t\tadown.contains( bup ) :\n\
\t\t\t\t\ta.compareDocumentPosition && a.compareDocumentPosition( bup ) & 16\n\
\t\t\t));\n\
\t\t} :\n\
\t\tfunction( a, b ) {\n\
\t\t\tif ( b ) {\n\
\t\t\t\twhile ( (b = b.parentNode) ) {\n\
\t\t\t\t\tif ( b === a ) {\n\
\t\t\t\t\t\treturn true;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t\treturn false;\n\
\t\t};\n\
\n\
\t/* Sorting\n\
\t---------------------------------------------------------------------- */\n\
\n\
\t// Document order sorting\n\
\tsortOrder = hasCompare ?\n\
\tfunction( a, b ) {\n\
\n\
\t\t// Flag for duplicate removal\n\
\t\tif ( a === b ) {\n\
\t\t\thasDuplicate = true;\n\
\t\t\treturn 0;\n\
\t\t}\n\
\n\
\t\t// Sort on method existence if only one input has compareDocumentPosition\n\
\t\tvar compare = !a.compareDocumentPosition - !b.compareDocumentPosition;\n\
\t\tif ( compare ) {\n\
\t\t\treturn compare;\n\
\t\t}\n\
\n\
\t\t// Calculate position if both inputs belong to the same document\n\
\t\tcompare = ( a.ownerDocument || a ) === ( b.ownerDocument || b ) ?\n\
\t\t\ta.compareDocumentPosition( b ) :\n\
\n\
\t\t\t// Otherwise we know they are disconnected\n\
\t\t\t1;\n\
\n\
\t\t// Disconnected nodes\n\
\t\tif ( compare & 1 ||\n\
\t\t\t(!support.sortDetached && b.compareDocumentPosition( a ) === compare) ) {\n\
\n\
\t\t\t// Choose the first element that is related to our preferred document\n\
\t\t\tif ( a === doc || a.ownerDocument === preferredDoc && contains(preferredDoc, a) ) {\n\
\t\t\t\treturn -1;\n\
\t\t\t}\n\
\t\t\tif ( b === doc || b.ownerDocument === preferredDoc && contains(preferredDoc, b) ) {\n\
\t\t\t\treturn 1;\n\
\t\t\t}\n\
\n\
\t\t\t// Maintain original order\n\
\t\t\treturn sortInput ?\n\
\t\t\t\t( indexOf.call( sortInput, a ) - indexOf.call( sortInput, b ) ) :\n\
\t\t\t\t0;\n\
\t\t}\n\
\n\
\t\treturn compare & 4 ? -1 : 1;\n\
\t} :\n\
\tfunction( a, b ) {\n\
\t\t// Exit early if the nodes are identical\n\
\t\tif ( a === b ) {\n\
\t\t\thasDuplicate = true;\n\
\t\t\treturn 0;\n\
\t\t}\n\
\n\
\t\tvar cur,\n\
\t\t\ti = 0,\n\
\t\t\taup = a.parentNode,\n\
\t\t\tbup = b.parentNode,\n\
\t\t\tap = [ a ],\n\
\t\t\tbp = [ b ];\n\
\n\
\t\t// Parentless nodes are either documents or disconnected\n\
\t\tif ( !aup || !bup ) {\n\
\t\t\treturn a === doc ? -1 :\n\
\t\t\t\tb === doc ? 1 :\n\
\t\t\t\taup ? -1 :\n\
\t\t\t\tbup ? 1 :\n\
\t\t\t\tsortInput ?\n\
\t\t\t\t( indexOf.call( sortInput, a ) - indexOf.call( sortInput, b ) ) :\n\
\t\t\t\t0;\n\
\n\
\t\t// If the nodes are siblings, we can do a quick check\n\
\t\t} else if ( aup === bup ) {\n\
\t\t\treturn siblingCheck( a, b );\n\
\t\t}\n\
\n\
\t\t// Otherwise we need full lists of their ancestors for comparison\n\
\t\tcur = a;\n\
\t\twhile ( (cur = cur.parentNode) ) {\n\
\t\t\tap.unshift( cur );\n\
\t\t}\n\
\t\tcur = b;\n\
\t\twhile ( (cur = cur.parentNode) ) {\n\
\t\t\tbp.unshift( cur );\n\
\t\t}\n\
\n\
\t\t// Walk down the tree looking for a discrepancy\n\
\t\twhile ( ap[i] === bp[i] ) {\n\
\t\t\ti++;\n\
\t\t}\n\
\n\
\t\treturn i ?\n\
\t\t\t// Do a sibling check if the nodes have a common ancestor\n\
\t\t\tsiblingCheck( ap[i], bp[i] ) :\n\
\n\
\t\t\t// Otherwise nodes in our document sort first\n\
\t\t\tap[i] === preferredDoc ? -1 :\n\
\t\t\tbp[i] === preferredDoc ? 1 :\n\
\t\t\t0;\n\
\t};\n\
\n\
\treturn doc;\n\
};\n\
\n\
Sizzle.matches = function( expr, elements ) {\n\
\treturn Sizzle( expr, null, null, elements );\n\
};\n\
\n\
Sizzle.matchesSelector = function( elem, expr ) {\n\
\t// Set document vars if needed\n\
\tif ( ( elem.ownerDocument || elem ) !== document ) {\n\
\t\tsetDocument( elem );\n\
\t}\n\
\n\
\t// Make sure that attribute selectors are quoted\n\
\texpr = expr.replace( rattributeQuotes, \"='$1']\" );\n\
\n\
\tif ( support.matchesSelector && documentIsHTML &&\n\
\t\t( !rbuggyMatches || !rbuggyMatches.test( expr ) ) &&\n\
\t\t( !rbuggyQSA     || !rbuggyQSA.test( expr ) ) ) {\n\
\n\
\t\ttry {\n\
\t\t\tvar ret = matches.call( elem, expr );\n\
\n\
\t\t\t// IE 9's matchesSelector returns false on disconnected nodes\n\
\t\t\tif ( ret || support.disconnectedMatch ||\n\
\t\t\t\t\t// As well, disconnected nodes are said to be in a document\n\
\t\t\t\t\t// fragment in IE 9\n\
\t\t\t\t\telem.document && elem.document.nodeType !== 11 ) {\n\
\t\t\t\treturn ret;\n\
\t\t\t}\n\
\t\t} catch(e) {}\n\
\t}\n\
\n\
\treturn Sizzle( expr, document, null, [ elem ] ).length > 0;\n\
};\n\
\n\
Sizzle.contains = function( context, elem ) {\n\
\t// Set document vars if needed\n\
\tif ( ( context.ownerDocument || context ) !== document ) {\n\
\t\tsetDocument( context );\n\
\t}\n\
\treturn contains( context, elem );\n\
};\n\
\n\
Sizzle.attr = function( elem, name ) {\n\
\t// Set document vars if needed\n\
\tif ( ( elem.ownerDocument || elem ) !== document ) {\n\
\t\tsetDocument( elem );\n\
\t}\n\
\n\
\tvar fn = Expr.attrHandle[ name.toLowerCase() ],\n\
\t\t// Don't get fooled by Object.prototype properties (jQuery #13807)\n\
\t\tval = fn && hasOwn.call( Expr.attrHandle, name.toLowerCase() ) ?\n\
\t\t\tfn( elem, name, !documentIsHTML ) :\n\
\t\t\tundefined;\n\
\n\
\treturn val !== undefined ?\n\
\t\tval :\n\
\t\tsupport.attributes || !documentIsHTML ?\n\
\t\t\telem.getAttribute( name ) :\n\
\t\t\t(val = elem.getAttributeNode(name)) && val.specified ?\n\
\t\t\t\tval.value :\n\
\t\t\t\tnull;\n\
};\n\
\n\
Sizzle.error = function( msg ) {\n\
\tthrow new Error( \"Syntax error, unrecognized expression: \" + msg );\n\
};\n\
\n\
/**\n\
 * Document sorting and removing duplicates\n\
 * @param {ArrayLike} results\n\
 */\n\
Sizzle.uniqueSort = function( results ) {\n\
\tvar elem,\n\
\t\tduplicates = [],\n\
\t\tj = 0,\n\
\t\ti = 0;\n\
\n\
\t// Unless we *know* we can detect duplicates, assume their presence\n\
\thasDuplicate = !support.detectDuplicates;\n\
\tsortInput = !support.sortStable && results.slice( 0 );\n\
\tresults.sort( sortOrder );\n\
\n\
\tif ( hasDuplicate ) {\n\
\t\twhile ( (elem = results[i++]) ) {\n\
\t\t\tif ( elem === results[ i ] ) {\n\
\t\t\t\tj = duplicates.push( i );\n\
\t\t\t}\n\
\t\t}\n\
\t\twhile ( j-- ) {\n\
\t\t\tresults.splice( duplicates[ j ], 1 );\n\
\t\t}\n\
\t}\n\
\n\
\t// Clear input after sorting to release objects\n\
\t// See https://github.com/jquery/sizzle/pull/225\n\
\tsortInput = null;\n\
\n\
\treturn results;\n\
};\n\
\n\
/**\n\
 * Utility function for retrieving the text value of an array of DOM nodes\n\
 * @param {Array|Element} elem\n\
 */\n\
getText = Sizzle.getText = function( elem ) {\n\
\tvar node,\n\
\t\tret = \"\",\n\
\t\ti = 0,\n\
\t\tnodeType = elem.nodeType;\n\
\n\
\tif ( !nodeType ) {\n\
\t\t// If no nodeType, this is expected to be an array\n\
\t\twhile ( (node = elem[i++]) ) {\n\
\t\t\t// Do not traverse comment nodes\n\
\t\t\tret += getText( node );\n\
\t\t}\n\
\t} else if ( nodeType === 1 || nodeType === 9 || nodeType === 11 ) {\n\
\t\t// Use textContent for elements\n\
\t\t// innerText usage removed for consistency of new lines (jQuery #11153)\n\
\t\tif ( typeof elem.textContent === \"string\" ) {\n\
\t\t\treturn elem.textContent;\n\
\t\t} else {\n\
\t\t\t// Traverse its children\n\
\t\t\tfor ( elem = elem.firstChild; elem; elem = elem.nextSibling ) {\n\
\t\t\t\tret += getText( elem );\n\
\t\t\t}\n\
\t\t}\n\
\t} else if ( nodeType === 3 || nodeType === 4 ) {\n\
\t\treturn elem.nodeValue;\n\
\t}\n\
\t// Do not include comment or processing instruction nodes\n\
\n\
\treturn ret;\n\
};\n\
\n\
Expr = Sizzle.selectors = {\n\
\n\
\t// Can be adjusted by the user\n\
\tcacheLength: 50,\n\
\n\
\tcreatePseudo: markFunction,\n\
\n\
\tmatch: matchExpr,\n\
\n\
\tattrHandle: {},\n\
\n\
\tfind: {},\n\
\n\
\trelative: {\n\
\t\t\">\": { dir: \"parentNode\", first: true },\n\
\t\t\" \": { dir: \"parentNode\" },\n\
\t\t\"+\": { dir: \"previousSibling\", first: true },\n\
\t\t\"~\": { dir: \"previousSibling\" }\n\
\t},\n\
\n\
\tpreFilter: {\n\
\t\t\"ATTR\": function( match ) {\n\
\t\t\tmatch[1] = match[1].replace( runescape, funescape );\n\
\n\
\t\t\t// Move the given value to match[3] whether quoted or unquoted\n\
\t\t\tmatch[3] = ( match[3] || match[4] || match[5] || \"\" ).replace( runescape, funescape );\n\
\n\
\t\t\tif ( match[2] === \"~=\" ) {\n\
\t\t\t\tmatch[3] = \" \" + match[3] + \" \";\n\
\t\t\t}\n\
\n\
\t\t\treturn match.slice( 0, 4 );\n\
\t\t},\n\
\n\
\t\t\"CHILD\": function( match ) {\n\
\t\t\t/* matches from matchExpr[\"CHILD\"]\n\
\t\t\t\t1 type (only|nth|...)\n\
\t\t\t\t2 what (child|of-type)\n\
\t\t\t\t3 argument (even|odd|\\d*|\\d*n([+-]\\d+)?|...)\n\
\t\t\t\t4 xn-component of xn+y argument ([+-]?\\d*n|)\n\
\t\t\t\t5 sign of xn-component\n\
\t\t\t\t6 x of xn-component\n\
\t\t\t\t7 sign of y-component\n\
\t\t\t\t8 y of y-component\n\
\t\t\t*/\n\
\t\t\tmatch[1] = match[1].toLowerCase();\n\
\n\
\t\t\tif ( match[1].slice( 0, 3 ) === \"nth\" ) {\n\
\t\t\t\t// nth-* requires argument\n\
\t\t\t\tif ( !match[3] ) {\n\
\t\t\t\t\tSizzle.error( match[0] );\n\
\t\t\t\t}\n\
\n\
\t\t\t\t// numeric x and y parameters for Expr.filter.CHILD\n\
\t\t\t\t// remember that false/true cast respectively to 0/1\n\
\t\t\t\tmatch[4] = +( match[4] ? match[5] + (match[6] || 1) : 2 * ( match[3] === \"even\" || match[3] === \"odd\" ) );\n\
\t\t\t\tmatch[5] = +( ( match[7] + match[8] ) || match[3] === \"odd\" );\n\
\n\
\t\t\t// other types prohibit arguments\n\
\t\t\t} else if ( match[3] ) {\n\
\t\t\t\tSizzle.error( match[0] );\n\
\t\t\t}\n\
\n\
\t\t\treturn match;\n\
\t\t},\n\
\n\
\t\t\"PSEUDO\": function( match ) {\n\
\t\t\tvar excess,\n\
\t\t\t\tunquoted = !match[6] && match[2];\n\
\n\
\t\t\tif ( matchExpr[\"CHILD\"].test( match[0] ) ) {\n\
\t\t\t\treturn null;\n\
\t\t\t}\n\
\n\
\t\t\t// Accept quoted arguments as-is\n\
\t\t\tif ( match[3] ) {\n\
\t\t\t\tmatch[2] = match[4] || match[5] || \"\";\n\
\n\
\t\t\t// Strip excess characters from unquoted arguments\n\
\t\t\t} else if ( unquoted && rpseudo.test( unquoted ) &&\n\
\t\t\t\t// Get excess from tokenize (recursively)\n\
\t\t\t\t(excess = tokenize( unquoted, true )) &&\n\
\t\t\t\t// advance to the next closing parenthesis\n\
\t\t\t\t(excess = unquoted.indexOf( \")\", unquoted.length - excess ) - unquoted.length) ) {\n\
\n\
\t\t\t\t// excess is a negative index\n\
\t\t\t\tmatch[0] = match[0].slice( 0, excess );\n\
\t\t\t\tmatch[2] = unquoted.slice( 0, excess );\n\
\t\t\t}\n\
\n\
\t\t\t// Return only captures needed by the pseudo filter method (type and argument)\n\
\t\t\treturn match.slice( 0, 3 );\n\
\t\t}\n\
\t},\n\
\n\
\tfilter: {\n\
\n\
\t\t\"TAG\": function( nodeNameSelector ) {\n\
\t\t\tvar nodeName = nodeNameSelector.replace( runescape, funescape ).toLowerCase();\n\
\t\t\treturn nodeNameSelector === \"*\" ?\n\
\t\t\t\tfunction() { return true; } :\n\
\t\t\t\tfunction( elem ) {\n\
\t\t\t\t\treturn elem.nodeName && elem.nodeName.toLowerCase() === nodeName;\n\
\t\t\t\t};\n\
\t\t},\n\
\n\
\t\t\"CLASS\": function( className ) {\n\
\t\t\tvar pattern = classCache[ className + \" \" ];\n\
\n\
\t\t\treturn pattern ||\n\
\t\t\t\t(pattern = new RegExp( \"(^|\" + whitespace + \")\" + className + \"(\" + whitespace + \"|$)\" )) &&\n\
\t\t\t\tclassCache( className, function( elem ) {\n\
\t\t\t\t\treturn pattern.test( typeof elem.className === \"string\" && elem.className || typeof elem.getAttribute !== strundefined && elem.getAttribute(\"class\") || \"\" );\n\
\t\t\t\t});\n\
\t\t},\n\
\n\
\t\t\"ATTR\": function( name, operator, check ) {\n\
\t\t\treturn function( elem ) {\n\
\t\t\t\tvar result = Sizzle.attr( elem, name );\n\
\n\
\t\t\t\tif ( result == null ) {\n\
\t\t\t\t\treturn operator === \"!=\";\n\
\t\t\t\t}\n\
\t\t\t\tif ( !operator ) {\n\
\t\t\t\t\treturn true;\n\
\t\t\t\t}\n\
\n\
\t\t\t\tresult += \"\";\n\
\n\
\t\t\t\treturn operator === \"=\" ? result === check :\n\
\t\t\t\t\toperator === \"!=\" ? result !== check :\n\
\t\t\t\t\toperator === \"^=\" ? check && result.indexOf( check ) === 0 :\n\
\t\t\t\t\toperator === \"*=\" ? check && result.indexOf( check ) > -1 :\n\
\t\t\t\t\toperator === \"$=\" ? check && result.slice( -check.length ) === check :\n\
\t\t\t\t\toperator === \"~=\" ? ( \" \" + result + \" \" ).indexOf( check ) > -1 :\n\
\t\t\t\t\toperator === \"|=\" ? result === check || result.slice( 0, check.length + 1 ) === check + \"-\" :\n\
\t\t\t\t\tfalse;\n\
\t\t\t};\n\
\t\t},\n\
\n\
\t\t\"CHILD\": function( type, what, argument, first, last ) {\n\
\t\t\tvar simple = type.slice( 0, 3 ) !== \"nth\",\n\
\t\t\t\tforward = type.slice( -4 ) !== \"last\",\n\
\t\t\t\tofType = what === \"of-type\";\n\
\n\
\t\t\treturn first === 1 && last === 0 ?\n\
\n\
\t\t\t\t// Shortcut for :nth-*(n)\n\
\t\t\t\tfunction( elem ) {\n\
\t\t\t\t\treturn !!elem.parentNode;\n\
\t\t\t\t} :\n\
\n\
\t\t\t\tfunction( elem, context, xml ) {\n\
\t\t\t\t\tvar cache, outerCache, node, diff, nodeIndex, start,\n\
\t\t\t\t\t\tdir = simple !== forward ? \"nextSibling\" : \"previousSibling\",\n\
\t\t\t\t\t\tparent = elem.parentNode,\n\
\t\t\t\t\t\tname = ofType && elem.nodeName.toLowerCase(),\n\
\t\t\t\t\t\tuseCache = !xml && !ofType;\n\
\n\
\t\t\t\t\tif ( parent ) {\n\
\n\
\t\t\t\t\t\t// :(first|last|only)-(child|of-type)\n\
\t\t\t\t\t\tif ( simple ) {\n\
\t\t\t\t\t\t\twhile ( dir ) {\n\
\t\t\t\t\t\t\t\tnode = elem;\n\
\t\t\t\t\t\t\t\twhile ( (node = node[ dir ]) ) {\n\
\t\t\t\t\t\t\t\t\tif ( ofType ? node.nodeName.toLowerCase() === name : node.nodeType === 1 ) {\n\
\t\t\t\t\t\t\t\t\t\treturn false;\n\
\t\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t\t// Reverse direction for :only-* (if we haven't yet done so)\n\
\t\t\t\t\t\t\t\tstart = dir = type === \"only\" && !start && \"nextSibling\";\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\treturn true;\n\
\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t\tstart = [ forward ? parent.firstChild : parent.lastChild ];\n\
\n\
\t\t\t\t\t\t// non-xml :nth-child(...) stores cache data on `parent`\n\
\t\t\t\t\t\tif ( forward && useCache ) {\n\
\t\t\t\t\t\t\t// Seek `elem` from a previously-cached index\n\
\t\t\t\t\t\t\touterCache = parent[ expando ] || (parent[ expando ] = {});\n\
\t\t\t\t\t\t\tcache = outerCache[ type ] || [];\n\
\t\t\t\t\t\t\tnodeIndex = cache[0] === dirruns && cache[1];\n\
\t\t\t\t\t\t\tdiff = cache[0] === dirruns && cache[2];\n\
\t\t\t\t\t\t\tnode = nodeIndex && parent.childNodes[ nodeIndex ];\n\
\n\
\t\t\t\t\t\t\twhile ( (node = ++nodeIndex && node && node[ dir ] ||\n\
\n\
\t\t\t\t\t\t\t\t// Fallback to seeking `elem` from the start\n\
\t\t\t\t\t\t\t\t(diff = nodeIndex = 0) || start.pop()) ) {\n\
\n\
\t\t\t\t\t\t\t\t// When found, cache indexes on `parent` and break\n\
\t\t\t\t\t\t\t\tif ( node.nodeType === 1 && ++diff && node === elem ) {\n\
\t\t\t\t\t\t\t\t\touterCache[ type ] = [ dirruns, nodeIndex, diff ];\n\
\t\t\t\t\t\t\t\t\tbreak;\n\
\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t\t// Use previously-cached element index if available\n\
\t\t\t\t\t\t} else if ( useCache && (cache = (elem[ expando ] || (elem[ expando ] = {}))[ type ]) && cache[0] === dirruns ) {\n\
\t\t\t\t\t\t\tdiff = cache[1];\n\
\n\
\t\t\t\t\t\t// xml :nth-child(...) or :nth-last-child(...) or :nth(-last)?-of-type(...)\n\
\t\t\t\t\t\t} else {\n\
\t\t\t\t\t\t\t// Use the same loop as above to seek `elem` from the start\n\
\t\t\t\t\t\t\twhile ( (node = ++nodeIndex && node && node[ dir ] ||\n\
\t\t\t\t\t\t\t\t(diff = nodeIndex = 0) || start.pop()) ) {\n\
\n\
\t\t\t\t\t\t\t\tif ( ( ofType ? node.nodeName.toLowerCase() === name : node.nodeType === 1 ) && ++diff ) {\n\
\t\t\t\t\t\t\t\t\t// Cache the index of each encountered element\n\
\t\t\t\t\t\t\t\t\tif ( useCache ) {\n\
\t\t\t\t\t\t\t\t\t\t(node[ expando ] || (node[ expando ] = {}))[ type ] = [ dirruns, diff ];\n\
\t\t\t\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t\t\t\t\tif ( node === elem ) {\n\
\t\t\t\t\t\t\t\t\t\tbreak;\n\
\t\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t\t// Incorporate the offset, then check against cycle size\n\
\t\t\t\t\t\tdiff -= last;\n\
\t\t\t\t\t\treturn diff === first || ( diff % first === 0 && diff / first >= 0 );\n\
\t\t\t\t\t}\n\
\t\t\t\t};\n\
\t\t},\n\
\n\
\t\t\"PSEUDO\": function( pseudo, argument ) {\n\
\t\t\t// pseudo-class names are case-insensitive\n\
\t\t\t// http://www.w3.org/TR/selectors/#pseudo-classes\n\
\t\t\t// Prioritize by case sensitivity in case custom pseudos are added with uppercase letters\n\
\t\t\t// Remember that setFilters inherits from pseudos\n\
\t\t\tvar args,\n\
\t\t\t\tfn = Expr.pseudos[ pseudo ] || Expr.setFilters[ pseudo.toLowerCase() ] ||\n\
\t\t\t\t\tSizzle.error( \"unsupported pseudo: \" + pseudo );\n\
\n\
\t\t\t// The user may use createPseudo to indicate that\n\
\t\t\t// arguments are needed to create the filter function\n\
\t\t\t// just as Sizzle does\n\
\t\t\tif ( fn[ expando ] ) {\n\
\t\t\t\treturn fn( argument );\n\
\t\t\t}\n\
\n\
\t\t\t// But maintain support for old signatures\n\
\t\t\tif ( fn.length > 1 ) {\n\
\t\t\t\targs = [ pseudo, pseudo, \"\", argument ];\n\
\t\t\t\treturn Expr.setFilters.hasOwnProperty( pseudo.toLowerCase() ) ?\n\
\t\t\t\t\tmarkFunction(function( seed, matches ) {\n\
\t\t\t\t\t\tvar idx,\n\
\t\t\t\t\t\t\tmatched = fn( seed, argument ),\n\
\t\t\t\t\t\t\ti = matched.length;\n\
\t\t\t\t\t\twhile ( i-- ) {\n\
\t\t\t\t\t\t\tidx = indexOf.call( seed, matched[i] );\n\
\t\t\t\t\t\t\tseed[ idx ] = !( matches[ idx ] = matched[i] );\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}) :\n\
\t\t\t\t\tfunction( elem ) {\n\
\t\t\t\t\t\treturn fn( elem, 0, args );\n\
\t\t\t\t\t};\n\
\t\t\t}\n\
\n\
\t\t\treturn fn;\n\
\t\t}\n\
\t},\n\
\n\
\tpseudos: {\n\
\t\t// Potentially complex pseudos\n\
\t\t\"not\": markFunction(function( selector ) {\n\
\t\t\t// Trim the selector passed to compile\n\
\t\t\t// to avoid treating leading and trailing\n\
\t\t\t// spaces as combinators\n\
\t\t\tvar input = [],\n\
\t\t\t\tresults = [],\n\
\t\t\t\tmatcher = compile( selector.replace( rtrim, \"$1\" ) );\n\
\n\
\t\t\treturn matcher[ expando ] ?\n\
\t\t\t\tmarkFunction(function( seed, matches, context, xml ) {\n\
\t\t\t\t\tvar elem,\n\
\t\t\t\t\t\tunmatched = matcher( seed, null, xml, [] ),\n\
\t\t\t\t\t\ti = seed.length;\n\
\n\
\t\t\t\t\t// Match elements unmatched by `matcher`\n\
\t\t\t\t\twhile ( i-- ) {\n\
\t\t\t\t\t\tif ( (elem = unmatched[i]) ) {\n\
\t\t\t\t\t\t\tseed[i] = !(matches[i] = elem);\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t}) :\n\
\t\t\t\tfunction( elem, context, xml ) {\n\
\t\t\t\t\tinput[0] = elem;\n\
\t\t\t\t\tmatcher( input, null, xml, results );\n\
\t\t\t\t\treturn !results.pop();\n\
\t\t\t\t};\n\
\t\t}),\n\
\n\
\t\t\"has\": markFunction(function( selector ) {\n\
\t\t\treturn function( elem ) {\n\
\t\t\t\treturn Sizzle( selector, elem ).length > 0;\n\
\t\t\t};\n\
\t\t}),\n\
\n\
\t\t\"contains\": markFunction(function( text ) {\n\
\t\t\treturn function( elem ) {\n\
\t\t\t\treturn ( elem.textContent || elem.innerText || getText( elem ) ).indexOf( text ) > -1;\n\
\t\t\t};\n\
\t\t}),\n\
\n\
\t\t// \"Whether an element is represented by a :lang() selector\n\
\t\t// is based solely on the element's language value\n\
\t\t// being equal to the identifier C,\n\
\t\t// or beginning with the identifier C immediately followed by \"-\".\n\
\t\t// The matching of C against the element's language value is performed case-insensitively.\n\
\t\t// The identifier C does not have to be a valid language name.\"\n\
\t\t// http://www.w3.org/TR/selectors/#lang-pseudo\n\
\t\t\"lang\": markFunction( function( lang ) {\n\
\t\t\t// lang value must be a valid identifier\n\
\t\t\tif ( !ridentifier.test(lang || \"\") ) {\n\
\t\t\t\tSizzle.error( \"unsupported lang: \" + lang );\n\
\t\t\t}\n\
\t\t\tlang = lang.replace( runescape, funescape ).toLowerCase();\n\
\t\t\treturn function( elem ) {\n\
\t\t\t\tvar elemLang;\n\
\t\t\t\tdo {\n\
\t\t\t\t\tif ( (elemLang = documentIsHTML ?\n\
\t\t\t\t\t\telem.lang :\n\
\t\t\t\t\t\telem.getAttribute(\"xml:lang\") || elem.getAttribute(\"lang\")) ) {\n\
\n\
\t\t\t\t\t\telemLang = elemLang.toLowerCase();\n\
\t\t\t\t\t\treturn elemLang === lang || elemLang.indexOf( lang + \"-\" ) === 0;\n\
\t\t\t\t\t}\n\
\t\t\t\t} while ( (elem = elem.parentNode) && elem.nodeType === 1 );\n\
\t\t\t\treturn false;\n\
\t\t\t};\n\
\t\t}),\n\
\n\
\t\t// Miscellaneous\n\
\t\t\"target\": function( elem ) {\n\
\t\t\tvar hash = window.location && window.location.hash;\n\
\t\t\treturn hash && hash.slice( 1 ) === elem.id;\n\
\t\t},\n\
\n\
\t\t\"root\": function( elem ) {\n\
\t\t\treturn elem === docElem;\n\
\t\t},\n\
\n\
\t\t\"focus\": function( elem ) {\n\
\t\t\treturn elem === document.activeElement && (!document.hasFocus || document.hasFocus()) && !!(elem.type || elem.href || ~elem.tabIndex);\n\
\t\t},\n\
\n\
\t\t// Boolean properties\n\
\t\t\"enabled\": function( elem ) {\n\
\t\t\treturn elem.disabled === false;\n\
\t\t},\n\
\n\
\t\t\"disabled\": function( elem ) {\n\
\t\t\treturn elem.disabled === true;\n\
\t\t},\n\
\n\
\t\t\"checked\": function( elem ) {\n\
\t\t\t// In CSS3, :checked should return both checked and selected elements\n\
\t\t\t// http://www.w3.org/TR/2011/REC-css3-selectors-20110929/#checked\n\
\t\t\tvar nodeName = elem.nodeName.toLowerCase();\n\
\t\t\treturn (nodeName === \"input\" && !!elem.checked) || (nodeName === \"option\" && !!elem.selected);\n\
\t\t},\n\
\n\
\t\t\"selected\": function( elem ) {\n\
\t\t\t// Accessing this property makes selected-by-default\n\
\t\t\t// options in Safari work properly\n\
\t\t\tif ( elem.parentNode ) {\n\
\t\t\t\telem.parentNode.selectedIndex;\n\
\t\t\t}\n\
\n\
\t\t\treturn elem.selected === true;\n\
\t\t},\n\
\n\
\t\t// Contents\n\
\t\t\"empty\": function( elem ) {\n\
\t\t\t// http://www.w3.org/TR/selectors/#empty-pseudo\n\
\t\t\t// :empty is negated by element (1) or content nodes (text: 3; cdata: 4; entity ref: 5),\n\
\t\t\t//   but not by others (comment: 8; processing instruction: 7; etc.)\n\
\t\t\t// nodeType < 6 works because attributes (2) do not appear as children\n\
\t\t\tfor ( elem = elem.firstChild; elem; elem = elem.nextSibling ) {\n\
\t\t\t\tif ( elem.nodeType < 6 ) {\n\
\t\t\t\t\treturn false;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t\treturn true;\n\
\t\t},\n\
\n\
\t\t\"parent\": function( elem ) {\n\
\t\t\treturn !Expr.pseudos[\"empty\"]( elem );\n\
\t\t},\n\
\n\
\t\t// Element/input types\n\
\t\t\"header\": function( elem ) {\n\
\t\t\treturn rheader.test( elem.nodeName );\n\
\t\t},\n\
\n\
\t\t\"input\": function( elem ) {\n\
\t\t\treturn rinputs.test( elem.nodeName );\n\
\t\t},\n\
\n\
\t\t\"button\": function( elem ) {\n\
\t\t\tvar name = elem.nodeName.toLowerCase();\n\
\t\t\treturn name === \"input\" && elem.type === \"button\" || name === \"button\";\n\
\t\t},\n\
\n\
\t\t\"text\": function( elem ) {\n\
\t\t\tvar attr;\n\
\t\t\treturn elem.nodeName.toLowerCase() === \"input\" &&\n\
\t\t\t\telem.type === \"text\" &&\n\
\n\
\t\t\t\t// Support: IE<8\n\
\t\t\t\t// New HTML5 attribute values (e.g., \"search\") appear with elem.type === \"text\"\n\
\t\t\t\t( (attr = elem.getAttribute(\"type\")) == null || attr.toLowerCase() === \"text\" );\n\
\t\t},\n\
\n\
\t\t// Position-in-collection\n\
\t\t\"first\": createPositionalPseudo(function() {\n\
\t\t\treturn [ 0 ];\n\
\t\t}),\n\
\n\
\t\t\"last\": createPositionalPseudo(function( matchIndexes, length ) {\n\
\t\t\treturn [ length - 1 ];\n\
\t\t}),\n\
\n\
\t\t\"eq\": createPositionalPseudo(function( matchIndexes, length, argument ) {\n\
\t\t\treturn [ argument < 0 ? argument + length : argument ];\n\
\t\t}),\n\
\n\
\t\t\"even\": createPositionalPseudo(function( matchIndexes, length ) {\n\
\t\t\tvar i = 0;\n\
\t\t\tfor ( ; i < length; i += 2 ) {\n\
\t\t\t\tmatchIndexes.push( i );\n\
\t\t\t}\n\
\t\t\treturn matchIndexes;\n\
\t\t}),\n\
\n\
\t\t\"odd\": createPositionalPseudo(function( matchIndexes, length ) {\n\
\t\t\tvar i = 1;\n\
\t\t\tfor ( ; i < length; i += 2 ) {\n\
\t\t\t\tmatchIndexes.push( i );\n\
\t\t\t}\n\
\t\t\treturn matchIndexes;\n\
\t\t}),\n\
\n\
\t\t\"lt\": createPositionalPseudo(function( matchIndexes, length, argument ) {\n\
\t\t\tvar i = argument < 0 ? argument + length : argument;\n\
\t\t\tfor ( ; --i >= 0; ) {\n\
\t\t\t\tmatchIndexes.push( i );\n\
\t\t\t}\n\
\t\t\treturn matchIndexes;\n\
\t\t}),\n\
\n\
\t\t\"gt\": createPositionalPseudo(function( matchIndexes, length, argument ) {\n\
\t\t\tvar i = argument < 0 ? argument + length : argument;\n\
\t\t\tfor ( ; ++i < length; ) {\n\
\t\t\t\tmatchIndexes.push( i );\n\
\t\t\t}\n\
\t\t\treturn matchIndexes;\n\
\t\t})\n\
\t}\n\
};\n\
\n\
Expr.pseudos[\"nth\"] = Expr.pseudos[\"eq\"];\n\
\n\
// Add button/input type pseudos\n\
for ( i in { radio: true, checkbox: true, file: true, password: true, image: true } ) {\n\
\tExpr.pseudos[ i ] = createInputPseudo( i );\n\
}\n\
for ( i in { submit: true, reset: true } ) {\n\
\tExpr.pseudos[ i ] = createButtonPseudo( i );\n\
}\n\
\n\
// Easy API for creating new setFilters\n\
function setFilters() {}\n\
setFilters.prototype = Expr.filters = Expr.pseudos;\n\
Expr.setFilters = new setFilters();\n\
\n\
tokenize = Sizzle.tokenize = function( selector, parseOnly ) {\n\
\tvar matched, match, tokens, type,\n\
\t\tsoFar, groups, preFilters,\n\
\t\tcached = tokenCache[ selector + \" \" ];\n\
\n\
\tif ( cached ) {\n\
\t\treturn parseOnly ? 0 : cached.slice( 0 );\n\
\t}\n\
\n\
\tsoFar = selector;\n\
\tgroups = [];\n\
\tpreFilters = Expr.preFilter;\n\
\n\
\twhile ( soFar ) {\n\
\n\
\t\t// Comma and first run\n\
\t\tif ( !matched || (match = rcomma.exec( soFar )) ) {\n\
\t\t\tif ( match ) {\n\
\t\t\t\t// Don't consume trailing commas as valid\n\
\t\t\t\tsoFar = soFar.slice( match[0].length ) || soFar;\n\
\t\t\t}\n\
\t\t\tgroups.push( (tokens = []) );\n\
\t\t}\n\
\n\
\t\tmatched = false;\n\
\n\
\t\t// Combinators\n\
\t\tif ( (match = rcombinators.exec( soFar )) ) {\n\
\t\t\tmatched = match.shift();\n\
\t\t\ttokens.push({\n\
\t\t\t\tvalue: matched,\n\
\t\t\t\t// Cast descendant combinators to space\n\
\t\t\t\ttype: match[0].replace( rtrim, \" \" )\n\
\t\t\t});\n\
\t\t\tsoFar = soFar.slice( matched.length );\n\
\t\t}\n\
\n\
\t\t// Filters\n\
\t\tfor ( type in Expr.filter ) {\n\
\t\t\tif ( (match = matchExpr[ type ].exec( soFar )) && (!preFilters[ type ] ||\n\
\t\t\t\t(match = preFilters[ type ]( match ))) ) {\n\
\t\t\t\tmatched = match.shift();\n\
\t\t\t\ttokens.push({\n\
\t\t\t\t\tvalue: matched,\n\
\t\t\t\t\ttype: type,\n\
\t\t\t\t\tmatches: match\n\
\t\t\t\t});\n\
\t\t\t\tsoFar = soFar.slice( matched.length );\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\tif ( !matched ) {\n\
\t\t\tbreak;\n\
\t\t}\n\
\t}\n\
\n\
\t// Return the length of the invalid excess\n\
\t// if we're just parsing\n\
\t// Otherwise, throw an error or return tokens\n\
\treturn parseOnly ?\n\
\t\tsoFar.length :\n\
\t\tsoFar ?\n\
\t\t\tSizzle.error( selector ) :\n\
\t\t\t// Cache the tokens\n\
\t\t\ttokenCache( selector, groups ).slice( 0 );\n\
};\n\
\n\
function toSelector( tokens ) {\n\
\tvar i = 0,\n\
\t\tlen = tokens.length,\n\
\t\tselector = \"\";\n\
\tfor ( ; i < len; i++ ) {\n\
\t\tselector += tokens[i].value;\n\
\t}\n\
\treturn selector;\n\
}\n\
\n\
function addCombinator( matcher, combinator, base ) {\n\
\tvar dir = combinator.dir,\n\
\t\tcheckNonElements = base && dir === \"parentNode\",\n\
\t\tdoneName = done++;\n\
\n\
\treturn combinator.first ?\n\
\t\t// Check against closest ancestor/preceding element\n\
\t\tfunction( elem, context, xml ) {\n\
\t\t\twhile ( (elem = elem[ dir ]) ) {\n\
\t\t\t\tif ( elem.nodeType === 1 || checkNonElements ) {\n\
\t\t\t\t\treturn matcher( elem, context, xml );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t} :\n\
\n\
\t\t// Check against all ancestor/preceding elements\n\
\t\tfunction( elem, context, xml ) {\n\
\t\t\tvar oldCache, outerCache,\n\
\t\t\t\tnewCache = [ dirruns, doneName ];\n\
\n\
\t\t\t// We can't set arbitrary data on XML nodes, so they don't benefit from dir caching\n\
\t\t\tif ( xml ) {\n\
\t\t\t\twhile ( (elem = elem[ dir ]) ) {\n\
\t\t\t\t\tif ( elem.nodeType === 1 || checkNonElements ) {\n\
\t\t\t\t\t\tif ( matcher( elem, context, xml ) ) {\n\
\t\t\t\t\t\t\treturn true;\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t} else {\n\
\t\t\t\twhile ( (elem = elem[ dir ]) ) {\n\
\t\t\t\t\tif ( elem.nodeType === 1 || checkNonElements ) {\n\
\t\t\t\t\t\touterCache = elem[ expando ] || (elem[ expando ] = {});\n\
\t\t\t\t\t\tif ( (oldCache = outerCache[ dir ]) &&\n\
\t\t\t\t\t\t\toldCache[ 0 ] === dirruns && oldCache[ 1 ] === doneName ) {\n\
\n\
\t\t\t\t\t\t\t// Assign to newCache so results back-propagate to previous elements\n\
\t\t\t\t\t\t\treturn (newCache[ 2 ] = oldCache[ 2 ]);\n\
\t\t\t\t\t\t} else {\n\
\t\t\t\t\t\t\t// Reuse newcache so results back-propagate to previous elements\n\
\t\t\t\t\t\t\touterCache[ dir ] = newCache;\n\
\n\
\t\t\t\t\t\t\t// A match means we're done; a fail means we have to keep checking\n\
\t\t\t\t\t\t\tif ( (newCache[ 2 ] = matcher( elem, context, xml )) ) {\n\
\t\t\t\t\t\t\t\treturn true;\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t};\n\
}\n\
\n\
function elementMatcher( matchers ) {\n\
\treturn matchers.length > 1 ?\n\
\t\tfunction( elem, context, xml ) {\n\
\t\t\tvar i = matchers.length;\n\
\t\t\twhile ( i-- ) {\n\
\t\t\t\tif ( !matchers[i]( elem, context, xml ) ) {\n\
\t\t\t\t\treturn false;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t\treturn true;\n\
\t\t} :\n\
\t\tmatchers[0];\n\
}\n\
\n\
function multipleContexts( selector, contexts, results ) {\n\
\tvar i = 0,\n\
\t\tlen = contexts.length;\n\
\tfor ( ; i < len; i++ ) {\n\
\t\tSizzle( selector, contexts[i], results );\n\
\t}\n\
\treturn results;\n\
}\n\
\n\
function condense( unmatched, map, filter, context, xml ) {\n\
\tvar elem,\n\
\t\tnewUnmatched = [],\n\
\t\ti = 0,\n\
\t\tlen = unmatched.length,\n\
\t\tmapped = map != null;\n\
\n\
\tfor ( ; i < len; i++ ) {\n\
\t\tif ( (elem = unmatched[i]) ) {\n\
\t\t\tif ( !filter || filter( elem, context, xml ) ) {\n\
\t\t\t\tnewUnmatched.push( elem );\n\
\t\t\t\tif ( mapped ) {\n\
\t\t\t\t\tmap.push( i );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\treturn newUnmatched;\n\
}\n\
\n\
function setMatcher( preFilter, selector, matcher, postFilter, postFinder, postSelector ) {\n\
\tif ( postFilter && !postFilter[ expando ] ) {\n\
\t\tpostFilter = setMatcher( postFilter );\n\
\t}\n\
\tif ( postFinder && !postFinder[ expando ] ) {\n\
\t\tpostFinder = setMatcher( postFinder, postSelector );\n\
\t}\n\
\treturn markFunction(function( seed, results, context, xml ) {\n\
\t\tvar temp, i, elem,\n\
\t\t\tpreMap = [],\n\
\t\t\tpostMap = [],\n\
\t\t\tpreexisting = results.length,\n\
\n\
\t\t\t// Get initial elements from seed or context\n\
\t\t\telems = seed || multipleContexts( selector || \"*\", context.nodeType ? [ context ] : context, [] ),\n\
\n\
\t\t\t// Prefilter to get matcher input, preserving a map for seed-results synchronization\n\
\t\t\tmatcherIn = preFilter && ( seed || !selector ) ?\n\
\t\t\t\tcondense( elems, preMap, preFilter, context, xml ) :\n\
\t\t\t\telems,\n\
\n\
\t\t\tmatcherOut = matcher ?\n\
\t\t\t\t// If we have a postFinder, or filtered seed, or non-seed postFilter or preexisting results,\n\
\t\t\t\tpostFinder || ( seed ? preFilter : preexisting || postFilter ) ?\n\
\n\
\t\t\t\t\t// ...intermediate processing is necessary\n\
\t\t\t\t\t[] :\n\
\n\
\t\t\t\t\t// ...otherwise use results directly\n\
\t\t\t\t\tresults :\n\
\t\t\t\tmatcherIn;\n\
\n\
\t\t// Find primary matches\n\
\t\tif ( matcher ) {\n\
\t\t\tmatcher( matcherIn, matcherOut, context, xml );\n\
\t\t}\n\
\n\
\t\t// Apply postFilter\n\
\t\tif ( postFilter ) {\n\
\t\t\ttemp = condense( matcherOut, postMap );\n\
\t\t\tpostFilter( temp, [], context, xml );\n\
\n\
\t\t\t// Un-match failing elements by moving them back to matcherIn\n\
\t\t\ti = temp.length;\n\
\t\t\twhile ( i-- ) {\n\
\t\t\t\tif ( (elem = temp[i]) ) {\n\
\t\t\t\t\tmatcherOut[ postMap[i] ] = !(matcherIn[ postMap[i] ] = elem);\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\tif ( seed ) {\n\
\t\t\tif ( postFinder || preFilter ) {\n\
\t\t\t\tif ( postFinder ) {\n\
\t\t\t\t\t// Get the final matcherOut by condensing this intermediate into postFinder contexts\n\
\t\t\t\t\ttemp = [];\n\
\t\t\t\t\ti = matcherOut.length;\n\
\t\t\t\t\twhile ( i-- ) {\n\
\t\t\t\t\t\tif ( (elem = matcherOut[i]) ) {\n\
\t\t\t\t\t\t\t// Restore matcherIn since elem is not yet a final match\n\
\t\t\t\t\t\t\ttemp.push( (matcherIn[i] = elem) );\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t\tpostFinder( null, (matcherOut = []), temp, xml );\n\
\t\t\t\t}\n\
\n\
\t\t\t\t// Move matched elements from seed to results to keep them synchronized\n\
\t\t\t\ti = matcherOut.length;\n\
\t\t\t\twhile ( i-- ) {\n\
\t\t\t\t\tif ( (elem = matcherOut[i]) &&\n\
\t\t\t\t\t\t(temp = postFinder ? indexOf.call( seed, elem ) : preMap[i]) > -1 ) {\n\
\n\
\t\t\t\t\t\tseed[temp] = !(results[temp] = elem);\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t// Add elements to results, through postFinder if defined\n\
\t\t} else {\n\
\t\t\tmatcherOut = condense(\n\
\t\t\t\tmatcherOut === results ?\n\
\t\t\t\t\tmatcherOut.splice( preexisting, matcherOut.length ) :\n\
\t\t\t\t\tmatcherOut\n\
\t\t\t);\n\
\t\t\tif ( postFinder ) {\n\
\t\t\t\tpostFinder( null, results, matcherOut, xml );\n\
\t\t\t} else {\n\
\t\t\t\tpush.apply( results, matcherOut );\n\
\t\t\t}\n\
\t\t}\n\
\t});\n\
}\n\
\n\
function matcherFromTokens( tokens ) {\n\
\tvar checkContext, matcher, j,\n\
\t\tlen = tokens.length,\n\
\t\tleadingRelative = Expr.relative[ tokens[0].type ],\n\
\t\timplicitRelative = leadingRelative || Expr.relative[\" \"],\n\
\t\ti = leadingRelative ? 1 : 0,\n\
\n\
\t\t// The foundational matcher ensures that elements are reachable from top-level context(s)\n\
\t\tmatchContext = addCombinator( function( elem ) {\n\
\t\t\treturn elem === checkContext;\n\
\t\t}, implicitRelative, true ),\n\
\t\tmatchAnyContext = addCombinator( function( elem ) {\n\
\t\t\treturn indexOf.call( checkContext, elem ) > -1;\n\
\t\t}, implicitRelative, true ),\n\
\t\tmatchers = [ function( elem, context, xml ) {\n\
\t\t\treturn ( !leadingRelative && ( xml || context !== outermostContext ) ) || (\n\
\t\t\t\t(checkContext = context).nodeType ?\n\
\t\t\t\t\tmatchContext( elem, context, xml ) :\n\
\t\t\t\t\tmatchAnyContext( elem, context, xml ) );\n\
\t\t} ];\n\
\n\
\tfor ( ; i < len; i++ ) {\n\
\t\tif ( (matcher = Expr.relative[ tokens[i].type ]) ) {\n\
\t\t\tmatchers = [ addCombinator(elementMatcher( matchers ), matcher) ];\n\
\t\t} else {\n\
\t\t\tmatcher = Expr.filter[ tokens[i].type ].apply( null, tokens[i].matches );\n\
\n\
\t\t\t// Return special upon seeing a positional matcher\n\
\t\t\tif ( matcher[ expando ] ) {\n\
\t\t\t\t// Find the next relative operator (if any) for proper handling\n\
\t\t\t\tj = ++i;\n\
\t\t\t\tfor ( ; j < len; j++ ) {\n\
\t\t\t\t\tif ( Expr.relative[ tokens[j].type ] ) {\n\
\t\t\t\t\t\tbreak;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t\treturn setMatcher(\n\
\t\t\t\t\ti > 1 && elementMatcher( matchers ),\n\
\t\t\t\t\ti > 1 && toSelector(\n\
\t\t\t\t\t\t// If the preceding token was a descendant combinator, insert an implicit any-element `*`\n\
\t\t\t\t\t\ttokens.slice( 0, i - 1 ).concat({ value: tokens[ i - 2 ].type === \" \" ? \"*\" : \"\" })\n\
\t\t\t\t\t).replace( rtrim, \"$1\" ),\n\
\t\t\t\t\tmatcher,\n\
\t\t\t\t\ti < j && matcherFromTokens( tokens.slice( i, j ) ),\n\
\t\t\t\t\tj < len && matcherFromTokens( (tokens = tokens.slice( j )) ),\n\
\t\t\t\t\tj < len && toSelector( tokens )\n\
\t\t\t\t);\n\
\t\t\t}\n\
\t\t\tmatchers.push( matcher );\n\
\t\t}\n\
\t}\n\
\n\
\treturn elementMatcher( matchers );\n\
}\n\
\n\
function matcherFromGroupMatchers( elementMatchers, setMatchers ) {\n\
\tvar bySet = setMatchers.length > 0,\n\
\t\tbyElement = elementMatchers.length > 0,\n\
\t\tsuperMatcher = function( seed, context, xml, results, outermost ) {\n\
\t\t\tvar elem, j, matcher,\n\
\t\t\t\tmatchedCount = 0,\n\
\t\t\t\ti = \"0\",\n\
\t\t\t\tunmatched = seed && [],\n\
\t\t\t\tsetMatched = [],\n\
\t\t\t\tcontextBackup = outermostContext,\n\
\t\t\t\t// We must always have either seed elements or outermost context\n\
\t\t\t\telems = seed || byElement && Expr.find[\"TAG\"]( \"*\", outermost ),\n\
\t\t\t\t// Use integer dirruns iff this is the outermost matcher\n\
\t\t\t\tdirrunsUnique = (dirruns += contextBackup == null ? 1 : Math.random() || 0.1),\n\
\t\t\t\tlen = elems.length;\n\
\n\
\t\t\tif ( outermost ) {\n\
\t\t\t\toutermostContext = context !== document && context;\n\
\t\t\t}\n\
\n\
\t\t\t// Add elements passing elementMatchers directly to results\n\
\t\t\t// Keep `i` a string if there are no elements so `matchedCount` will be \"00\" below\n\
\t\t\t// Support: IE<9, Safari\n\
\t\t\t// Tolerate NodeList properties (IE: \"length\"; Safari: <number>) matching elements by id\n\
\t\t\tfor ( ; i !== len && (elem = elems[i]) != null; i++ ) {\n\
\t\t\t\tif ( byElement && elem ) {\n\
\t\t\t\t\tj = 0;\n\
\t\t\t\t\twhile ( (matcher = elementMatchers[j++]) ) {\n\
\t\t\t\t\t\tif ( matcher( elem, context, xml ) ) {\n\
\t\t\t\t\t\t\tresults.push( elem );\n\
\t\t\t\t\t\t\tbreak;\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t\tif ( outermost ) {\n\
\t\t\t\t\t\tdirruns = dirrunsUnique;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\n\
\t\t\t\t// Track unmatched elements for set filters\n\
\t\t\t\tif ( bySet ) {\n\
\t\t\t\t\t// They will have gone through all possible matchers\n\
\t\t\t\t\tif ( (elem = !matcher && elem) ) {\n\
\t\t\t\t\t\tmatchedCount--;\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// Lengthen the array for every element, matched or not\n\
\t\t\t\t\tif ( seed ) {\n\
\t\t\t\t\t\tunmatched.push( elem );\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t\t// Apply set filters to unmatched elements\n\
\t\t\tmatchedCount += i;\n\
\t\t\tif ( bySet && i !== matchedCount ) {\n\
\t\t\t\tj = 0;\n\
\t\t\t\twhile ( (matcher = setMatchers[j++]) ) {\n\
\t\t\t\t\tmatcher( unmatched, setMatched, context, xml );\n\
\t\t\t\t}\n\
\n\
\t\t\t\tif ( seed ) {\n\
\t\t\t\t\t// Reintegrate element matches to eliminate the need for sorting\n\
\t\t\t\t\tif ( matchedCount > 0 ) {\n\
\t\t\t\t\t\twhile ( i-- ) {\n\
\t\t\t\t\t\t\tif ( !(unmatched[i] || setMatched[i]) ) {\n\
\t\t\t\t\t\t\t\tsetMatched[i] = pop.call( results );\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// Discard index placeholder values to get only actual matches\n\
\t\t\t\t\tsetMatched = condense( setMatched );\n\
\t\t\t\t}\n\
\n\
\t\t\t\t// Add matches to results\n\
\t\t\t\tpush.apply( results, setMatched );\n\
\n\
\t\t\t\t// Seedless set matches succeeding multiple successful matchers stipulate sorting\n\
\t\t\t\tif ( outermost && !seed && setMatched.length > 0 &&\n\
\t\t\t\t\t( matchedCount + setMatchers.length ) > 1 ) {\n\
\n\
\t\t\t\t\tSizzle.uniqueSort( results );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t\t// Override manipulation of globals by nested matchers\n\
\t\t\tif ( outermost ) {\n\
\t\t\t\tdirruns = dirrunsUnique;\n\
\t\t\t\toutermostContext = contextBackup;\n\
\t\t\t}\n\
\n\
\t\t\treturn unmatched;\n\
\t\t};\n\
\n\
\treturn bySet ?\n\
\t\tmarkFunction( superMatcher ) :\n\
\t\tsuperMatcher;\n\
}\n\
\n\
compile = Sizzle.compile = function( selector, match /* Internal Use Only */ ) {\n\
\tvar i,\n\
\t\tsetMatchers = [],\n\
\t\telementMatchers = [],\n\
\t\tcached = compilerCache[ selector + \" \" ];\n\
\n\
\tif ( !cached ) {\n\
\t\t// Generate a function of recursive functions that can be used to check each element\n\
\t\tif ( !match ) {\n\
\t\t\tmatch = tokenize( selector );\n\
\t\t}\n\
\t\ti = match.length;\n\
\t\twhile ( i-- ) {\n\
\t\t\tcached = matcherFromTokens( match[i] );\n\
\t\t\tif ( cached[ expando ] ) {\n\
\t\t\t\tsetMatchers.push( cached );\n\
\t\t\t} else {\n\
\t\t\t\telementMatchers.push( cached );\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Cache the compiled function\n\
\t\tcached = compilerCache( selector, matcherFromGroupMatchers( elementMatchers, setMatchers ) );\n\
\n\
\t\t// Save selector and tokenization\n\
\t\tcached.selector = selector;\n\
\t}\n\
\treturn cached;\n\
};\n\
\n\
/**\n\
 * A low-level selection function that works with Sizzle's compiled\n\
 *  selector functions\n\
 * @param {String|Function} selector A selector or a pre-compiled\n\
 *  selector function built with Sizzle.compile\n\
 * @param {Element} context\n\
 * @param {Array} [results]\n\
 * @param {Array} [seed] A set of elements to match against\n\
 */\n\
select = Sizzle.select = function( selector, context, results, seed ) {\n\
\tvar i, tokens, token, type, find,\n\
\t\tcompiled = typeof selector === \"function\" && selector,\n\
\t\tmatch = !seed && tokenize( (selector = compiled.selector || selector) );\n\
\n\
\tresults = results || [];\n\
\n\
\t// Try to minimize operations if there is no seed and only one group\n\
\tif ( match.length === 1 ) {\n\
\n\
\t\t// Take a shortcut and set the context if the root selector is an ID\n\
\t\ttokens = match[0] = match[0].slice( 0 );\n\
\t\tif ( tokens.length > 2 && (token = tokens[0]).type === \"ID\" &&\n\
\t\t\t\tsupport.getById && context.nodeType === 9 && documentIsHTML &&\n\
\t\t\t\tExpr.relative[ tokens[1].type ] ) {\n\
\n\
\t\t\tcontext = ( Expr.find[\"ID\"]( token.matches[0].replace(runescape, funescape), context ) || [] )[0];\n\
\t\t\tif ( !context ) {\n\
\t\t\t\treturn results;\n\
\n\
\t\t\t// Precompiled matchers will still verify ancestry, so step up a level\n\
\t\t\t} else if ( compiled ) {\n\
\t\t\t\tcontext = context.parentNode;\n\
\t\t\t}\n\
\n\
\t\t\tselector = selector.slice( tokens.shift().value.length );\n\
\t\t}\n\
\n\
\t\t// Fetch a seed set for right-to-left matching\n\
\t\ti = matchExpr[\"needsContext\"].test( selector ) ? 0 : tokens.length;\n\
\t\twhile ( i-- ) {\n\
\t\t\ttoken = tokens[i];\n\
\n\
\t\t\t// Abort if we hit a combinator\n\
\t\t\tif ( Expr.relative[ (type = token.type) ] ) {\n\
\t\t\t\tbreak;\n\
\t\t\t}\n\
\t\t\tif ( (find = Expr.find[ type ]) ) {\n\
\t\t\t\t// Search, expanding context for leading sibling combinators\n\
\t\t\t\tif ( (seed = find(\n\
\t\t\t\t\ttoken.matches[0].replace( runescape, funescape ),\n\
\t\t\t\t\trsibling.test( tokens[0].type ) && testContext( context.parentNode ) || context\n\
\t\t\t\t)) ) {\n\
\n\
\t\t\t\t\t// If seed is empty or no tokens remain, we can return early\n\
\t\t\t\t\ttokens.splice( i, 1 );\n\
\t\t\t\t\tselector = seed.length && toSelector( tokens );\n\
\t\t\t\t\tif ( !selector ) {\n\
\t\t\t\t\t\tpush.apply( results, seed );\n\
\t\t\t\t\t\treturn results;\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\tbreak;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\t// Compile and execute a filtering function if one is not provided\n\
\t// Provide `match` to avoid retokenization if we modified the selector above\n\
\t( compiled || compile( selector, match ) )(\n\
\t\tseed,\n\
\t\tcontext,\n\
\t\t!documentIsHTML,\n\
\t\tresults,\n\
\t\trsibling.test( selector ) && testContext( context.parentNode ) || context\n\
\t);\n\
\treturn results;\n\
};\n\
\n\
// One-time assignments\n\
\n\
// Sort stability\n\
support.sortStable = expando.split(\"\").sort( sortOrder ).join(\"\") === expando;\n\
\n\
// Support: Chrome<14\n\
// Always assume duplicates if they aren't passed to the comparison function\n\
support.detectDuplicates = !!hasDuplicate;\n\
\n\
// Initialize against the default document\n\
setDocument();\n\
\n\
// Support: Webkit<537.32 - Safari 6.0.3/Chrome 25 (fixed in Chrome 27)\n\
// Detached nodes confoundingly follow *each other*\n\
support.sortDetached = assert(function( div1 ) {\n\
\t// Should return 1, but returns 4 (following)\n\
\treturn div1.compareDocumentPosition( document.createElement(\"div\") ) & 1;\n\
});\n\
\n\
// Support: IE<8\n\
// Prevent attribute/property \"interpolation\"\n\
// http://msdn.microsoft.com/en-us/library/ms536429%28VS.85%29.aspx\n\
if ( !assert(function( div ) {\n\
\tdiv.innerHTML = \"<a href='#'></a>\";\n\
\treturn div.firstChild.getAttribute(\"href\") === \"#\" ;\n\
}) ) {\n\
\taddHandle( \"type|href|height|width\", function( elem, name, isXML ) {\n\
\t\tif ( !isXML ) {\n\
\t\t\treturn elem.getAttribute( name, name.toLowerCase() === \"type\" ? 1 : 2 );\n\
\t\t}\n\
\t});\n\
}\n\
\n\
// Support: IE<9\n\
// Use defaultValue in place of getAttribute(\"value\")\n\
if ( !support.attributes || !assert(function( div ) {\n\
\tdiv.innerHTML = \"<input/>\";\n\
\tdiv.firstChild.setAttribute( \"value\", \"\" );\n\
\treturn div.firstChild.getAttribute( \"value\" ) === \"\";\n\
}) ) {\n\
\taddHandle( \"value\", function( elem, name, isXML ) {\n\
\t\tif ( !isXML && elem.nodeName.toLowerCase() === \"input\" ) {\n\
\t\t\treturn elem.defaultValue;\n\
\t\t}\n\
\t});\n\
}\n\
\n\
// Support: IE<9\n\
// Use getAttributeNode to fetch booleans when getAttribute lies\n\
if ( !assert(function( div ) {\n\
\treturn div.getAttribute(\"disabled\") == null;\n\
}) ) {\n\
\taddHandle( booleans, function( elem, name, isXML ) {\n\
\t\tvar val;\n\
\t\tif ( !isXML ) {\n\
\t\t\treturn elem[ name ] === true ? name.toLowerCase() :\n\
\t\t\t\t\t(val = elem.getAttributeNode( name )) && val.specified ?\n\
\t\t\t\t\tval.value :\n\
\t\t\t\tnull;\n\
\t\t}\n\
\t});\n\
}\n\
\n\
return Sizzle;\n\
\n\
})( window );\n\
\n\
\n\
\n\
jQuery.find = Sizzle;\n\
jQuery.expr = Sizzle.selectors;\n\
jQuery.expr[\":\"] = jQuery.expr.pseudos;\n\
jQuery.unique = Sizzle.uniqueSort;\n\
jQuery.text = Sizzle.getText;\n\
jQuery.isXMLDoc = Sizzle.isXML;\n\
jQuery.contains = Sizzle.contains;\n\
\n\
\n\
\n\
var rneedsContext = jQuery.expr.match.needsContext;\n\
\n\
var rsingleTag = (/^<(\\w+)\\s*\\/?>(?:<\\/\\1>|)$/);\n\
\n\
\n\
\n\
var risSimple = /^.[^:#\\[\\.,]*$/;\n\
\n\
// Implement the identical functionality for filter and not\n\
function winnow( elements, qualifier, not ) {\n\
\tif ( jQuery.isFunction( qualifier ) ) {\n\
\t\treturn jQuery.grep( elements, function( elem, i ) {\n\
\t\t\t/* jshint -W018 */\n\
\t\t\treturn !!qualifier.call( elem, i, elem ) !== not;\n\
\t\t});\n\
\n\
\t}\n\
\n\
\tif ( qualifier.nodeType ) {\n\
\t\treturn jQuery.grep( elements, function( elem ) {\n\
\t\t\treturn ( elem === qualifier ) !== not;\n\
\t\t});\n\
\n\
\t}\n\
\n\
\tif ( typeof qualifier === \"string\" ) {\n\
\t\tif ( risSimple.test( qualifier ) ) {\n\
\t\t\treturn jQuery.filter( qualifier, elements, not );\n\
\t\t}\n\
\n\
\t\tqualifier = jQuery.filter( qualifier, elements );\n\
\t}\n\
\n\
\treturn jQuery.grep( elements, function( elem ) {\n\
\t\treturn ( jQuery.inArray( elem, qualifier ) >= 0 ) !== not;\n\
\t});\n\
}\n\
\n\
jQuery.filter = function( expr, elems, not ) {\n\
\tvar elem = elems[ 0 ];\n\
\n\
\tif ( not ) {\n\
\t\texpr = \":not(\" + expr + \")\";\n\
\t}\n\
\n\
\treturn elems.length === 1 && elem.nodeType === 1 ?\n\
\t\tjQuery.find.matchesSelector( elem, expr ) ? [ elem ] : [] :\n\
\t\tjQuery.find.matches( expr, jQuery.grep( elems, function( elem ) {\n\
\t\t\treturn elem.nodeType === 1;\n\
\t\t}));\n\
};\n\
\n\
jQuery.fn.extend({\n\
\tfind: function( selector ) {\n\
\t\tvar i,\n\
\t\t\tret = [],\n\
\t\t\tself = this,\n\
\t\t\tlen = self.length;\n\
\n\
\t\tif ( typeof selector !== \"string\" ) {\n\
\t\t\treturn this.pushStack( jQuery( selector ).filter(function() {\n\
\t\t\t\tfor ( i = 0; i < len; i++ ) {\n\
\t\t\t\t\tif ( jQuery.contains( self[ i ], this ) ) {\n\
\t\t\t\t\t\treturn true;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}) );\n\
\t\t}\n\
\n\
\t\tfor ( i = 0; i < len; i++ ) {\n\
\t\t\tjQuery.find( selector, self[ i ], ret );\n\
\t\t}\n\
\n\
\t\t// Needed because $( selector, context ) becomes $( context ).find( selector )\n\
\t\tret = this.pushStack( len > 1 ? jQuery.unique( ret ) : ret );\n\
\t\tret.selector = this.selector ? this.selector + \" \" + selector : selector;\n\
\t\treturn ret;\n\
\t},\n\
\tfilter: function( selector ) {\n\
\t\treturn this.pushStack( winnow(this, selector || [], false) );\n\
\t},\n\
\tnot: function( selector ) {\n\
\t\treturn this.pushStack( winnow(this, selector || [], true) );\n\
\t},\n\
\tis: function( selector ) {\n\
\t\treturn !!winnow(\n\
\t\t\tthis,\n\
\n\
\t\t\t// If this is a positional/relative selector, check membership in the returned set\n\
\t\t\t// so $(\"p:first\").is(\"p:last\") won't return true for a doc with two \"p\".\n\
\t\t\ttypeof selector === \"string\" && rneedsContext.test( selector ) ?\n\
\t\t\t\tjQuery( selector ) :\n\
\t\t\t\tselector || [],\n\
\t\t\tfalse\n\
\t\t).length;\n\
\t}\n\
});\n\
\n\
\n\
// Initialize a jQuery object\n\
\n\
\n\
// A central reference to the root jQuery(document)\n\
var rootjQuery,\n\
\n\
\t// Use the correct document accordingly with window argument (sandbox)\n\
\tdocument = window.document,\n\
\n\
\t// A simple way to check for HTML strings\n\
\t// Prioritize #id over <tag> to avoid XSS via location.hash (#9521)\n\
\t// Strict HTML recognition (#11290: must start with <)\n\
\trquickExpr = /^(?:\\s*(<[\\w\\W]+>)[^>]*|#([\\w-]*))$/,\n\
\n\
\tinit = jQuery.fn.init = function( selector, context ) {\n\
\t\tvar match, elem;\n\
\n\
\t\t// HANDLE: $(\"\"), $(null), $(undefined), $(false)\n\
\t\tif ( !selector ) {\n\
\t\t\treturn this;\n\
\t\t}\n\
\n\
\t\t// Handle HTML strings\n\
\t\tif ( typeof selector === \"string\" ) {\n\
\t\t\tif ( selector.charAt(0) === \"<\" && selector.charAt( selector.length - 1 ) === \">\" && selector.length >= 3 ) {\n\
\t\t\t\t// Assume that strings that start and end with <> are HTML and skip the regex check\n\
\t\t\t\tmatch = [ null, selector, null ];\n\
\n\
\t\t\t} else {\n\
\t\t\t\tmatch = rquickExpr.exec( selector );\n\
\t\t\t}\n\
\n\
\t\t\t// Match html or make sure no context is specified for #id\n\
\t\t\tif ( match && (match[1] || !context) ) {\n\
\n\
\t\t\t\t// HANDLE: $(html) -> $(array)\n\
\t\t\t\tif ( match[1] ) {\n\
\t\t\t\t\tcontext = context instanceof jQuery ? context[0] : context;\n\
\n\
\t\t\t\t\t// scripts is true for back-compat\n\
\t\t\t\t\t// Intentionally let the error be thrown if parseHTML is not present\n\
\t\t\t\t\tjQuery.merge( this, jQuery.parseHTML(\n\
\t\t\t\t\t\tmatch[1],\n\
\t\t\t\t\t\tcontext && context.nodeType ? context.ownerDocument || context : document,\n\
\t\t\t\t\t\ttrue\n\
\t\t\t\t\t) );\n\
\n\
\t\t\t\t\t// HANDLE: $(html, props)\n\
\t\t\t\t\tif ( rsingleTag.test( match[1] ) && jQuery.isPlainObject( context ) ) {\n\
\t\t\t\t\t\tfor ( match in context ) {\n\
\t\t\t\t\t\t\t// Properties of context are called as methods if possible\n\
\t\t\t\t\t\t\tif ( jQuery.isFunction( this[ match ] ) ) {\n\
\t\t\t\t\t\t\t\tthis[ match ]( context[ match ] );\n\
\n\
\t\t\t\t\t\t\t// ...and otherwise set as attributes\n\
\t\t\t\t\t\t\t} else {\n\
\t\t\t\t\t\t\t\tthis.attr( match, context[ match ] );\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\treturn this;\n\
\n\
\t\t\t\t// HANDLE: $(#id)\n\
\t\t\t\t} else {\n\
\t\t\t\t\telem = document.getElementById( match[2] );\n\
\n\
\t\t\t\t\t// Check parentNode to catch when Blackberry 4.6 returns\n\
\t\t\t\t\t// nodes that are no longer in the document #6963\n\
\t\t\t\t\tif ( elem && elem.parentNode ) {\n\
\t\t\t\t\t\t// Handle the case where IE and Opera return items\n\
\t\t\t\t\t\t// by name instead of ID\n\
\t\t\t\t\t\tif ( elem.id !== match[2] ) {\n\
\t\t\t\t\t\t\treturn rootjQuery.find( selector );\n\
\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t\t// Otherwise, we inject the element directly into the jQuery object\n\
\t\t\t\t\t\tthis.length = 1;\n\
\t\t\t\t\t\tthis[0] = elem;\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\tthis.context = document;\n\
\t\t\t\t\tthis.selector = selector;\n\
\t\t\t\t\treturn this;\n\
\t\t\t\t}\n\
\n\
\t\t\t// HANDLE: $(expr, $(...))\n\
\t\t\t} else if ( !context || context.jquery ) {\n\
\t\t\t\treturn ( context || rootjQuery ).find( selector );\n\
\n\
\t\t\t// HANDLE: $(expr, context)\n\
\t\t\t// (which is just equivalent to: $(context).find(expr)\n\
\t\t\t} else {\n\
\t\t\t\treturn this.constructor( context ).find( selector );\n\
\t\t\t}\n\
\n\
\t\t// HANDLE: $(DOMElement)\n\
\t\t} else if ( selector.nodeType ) {\n\
\t\t\tthis.context = this[0] = selector;\n\
\t\t\tthis.length = 1;\n\
\t\t\treturn this;\n\
\n\
\t\t// HANDLE: $(function)\n\
\t\t// Shortcut for document ready\n\
\t\t} else if ( jQuery.isFunction( selector ) ) {\n\
\t\t\treturn typeof rootjQuery.ready !== \"undefined\" ?\n\
\t\t\t\trootjQuery.ready( selector ) :\n\
\t\t\t\t// Execute immediately if ready is not present\n\
\t\t\t\tselector( jQuery );\n\
\t\t}\n\
\n\
\t\tif ( selector.selector !== undefined ) {\n\
\t\t\tthis.selector = selector.selector;\n\
\t\t\tthis.context = selector.context;\n\
\t\t}\n\
\n\
\t\treturn jQuery.makeArray( selector, this );\n\
\t};\n\
\n\
// Give the init function the jQuery prototype for later instantiation\n\
init.prototype = jQuery.fn;\n\
\n\
// Initialize central reference\n\
rootjQuery = jQuery( document );\n\
\n\
\n\
var rparentsprev = /^(?:parents|prev(?:Until|All))/,\n\
\t// methods guaranteed to produce a unique set when starting from a unique set\n\
\tguaranteedUnique = {\n\
\t\tchildren: true,\n\
\t\tcontents: true,\n\
\t\tnext: true,\n\
\t\tprev: true\n\
\t};\n\
\n\
jQuery.extend({\n\
\tdir: function( elem, dir, until ) {\n\
\t\tvar matched = [],\n\
\t\t\tcur = elem[ dir ];\n\
\n\
\t\twhile ( cur && cur.nodeType !== 9 && (until === undefined || cur.nodeType !== 1 || !jQuery( cur ).is( until )) ) {\n\
\t\t\tif ( cur.nodeType === 1 ) {\n\
\t\t\t\tmatched.push( cur );\n\
\t\t\t}\n\
\t\t\tcur = cur[dir];\n\
\t\t}\n\
\t\treturn matched;\n\
\t},\n\
\n\
\tsibling: function( n, elem ) {\n\
\t\tvar r = [];\n\
\n\
\t\tfor ( ; n; n = n.nextSibling ) {\n\
\t\t\tif ( n.nodeType === 1 && n !== elem ) {\n\
\t\t\t\tr.push( n );\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn r;\n\
\t}\n\
});\n\
\n\
jQuery.fn.extend({\n\
\thas: function( target ) {\n\
\t\tvar i,\n\
\t\t\ttargets = jQuery( target, this ),\n\
\t\t\tlen = targets.length;\n\
\n\
\t\treturn this.filter(function() {\n\
\t\t\tfor ( i = 0; i < len; i++ ) {\n\
\t\t\t\tif ( jQuery.contains( this, targets[i] ) ) {\n\
\t\t\t\t\treturn true;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t});\n\
\t},\n\
\n\
\tclosest: function( selectors, context ) {\n\
\t\tvar cur,\n\
\t\t\ti = 0,\n\
\t\t\tl = this.length,\n\
\t\t\tmatched = [],\n\
\t\t\tpos = rneedsContext.test( selectors ) || typeof selectors !== \"string\" ?\n\
\t\t\t\tjQuery( selectors, context || this.context ) :\n\
\t\t\t\t0;\n\
\n\
\t\tfor ( ; i < l; i++ ) {\n\
\t\t\tfor ( cur = this[i]; cur && cur !== context; cur = cur.parentNode ) {\n\
\t\t\t\t// Always skip document fragments\n\
\t\t\t\tif ( cur.nodeType < 11 && (pos ?\n\
\t\t\t\t\tpos.index(cur) > -1 :\n\
\n\
\t\t\t\t\t// Don't pass non-elements to Sizzle\n\
\t\t\t\t\tcur.nodeType === 1 &&\n\
\t\t\t\t\t\tjQuery.find.matchesSelector(cur, selectors)) ) {\n\
\n\
\t\t\t\t\tmatched.push( cur );\n\
\t\t\t\t\tbreak;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn this.pushStack( matched.length > 1 ? jQuery.unique( matched ) : matched );\n\
\t},\n\
\n\
\t// Determine the position of an element within\n\
\t// the matched set of elements\n\
\tindex: function( elem ) {\n\
\n\
\t\t// No argument, return index in parent\n\
\t\tif ( !elem ) {\n\
\t\t\treturn ( this[0] && this[0].parentNode ) ? this.first().prevAll().length : -1;\n\
\t\t}\n\
\n\
\t\t// index in selector\n\
\t\tif ( typeof elem === \"string\" ) {\n\
\t\t\treturn jQuery.inArray( this[0], jQuery( elem ) );\n\
\t\t}\n\
\n\
\t\t// Locate the position of the desired element\n\
\t\treturn jQuery.inArray(\n\
\t\t\t// If it receives a jQuery object, the first element is used\n\
\t\t\telem.jquery ? elem[0] : elem, this );\n\
\t},\n\
\n\
\tadd: function( selector, context ) {\n\
\t\treturn this.pushStack(\n\
\t\t\tjQuery.unique(\n\
\t\t\t\tjQuery.merge( this.get(), jQuery( selector, context ) )\n\
\t\t\t)\n\
\t\t);\n\
\t},\n\
\n\
\taddBack: function( selector ) {\n\
\t\treturn this.add( selector == null ?\n\
\t\t\tthis.prevObject : this.prevObject.filter(selector)\n\
\t\t);\n\
\t}\n\
});\n\
\n\
function sibling( cur, dir ) {\n\
\tdo {\n\
\t\tcur = cur[ dir ];\n\
\t} while ( cur && cur.nodeType !== 1 );\n\
\n\
\treturn cur;\n\
}\n\
\n\
jQuery.each({\n\
\tparent: function( elem ) {\n\
\t\tvar parent = elem.parentNode;\n\
\t\treturn parent && parent.nodeType !== 11 ? parent : null;\n\
\t},\n\
\tparents: function( elem ) {\n\
\t\treturn jQuery.dir( elem, \"parentNode\" );\n\
\t},\n\
\tparentsUntil: function( elem, i, until ) {\n\
\t\treturn jQuery.dir( elem, \"parentNode\", until );\n\
\t},\n\
\tnext: function( elem ) {\n\
\t\treturn sibling( elem, \"nextSibling\" );\n\
\t},\n\
\tprev: function( elem ) {\n\
\t\treturn sibling( elem, \"previousSibling\" );\n\
\t},\n\
\tnextAll: function( elem ) {\n\
\t\treturn jQuery.dir( elem, \"nextSibling\" );\n\
\t},\n\
\tprevAll: function( elem ) {\n\
\t\treturn jQuery.dir( elem, \"previousSibling\" );\n\
\t},\n\
\tnextUntil: function( elem, i, until ) {\n\
\t\treturn jQuery.dir( elem, \"nextSibling\", until );\n\
\t},\n\
\tprevUntil: function( elem, i, until ) {\n\
\t\treturn jQuery.dir( elem, \"previousSibling\", until );\n\
\t},\n\
\tsiblings: function( elem ) {\n\
\t\treturn jQuery.sibling( ( elem.parentNode || {} ).firstChild, elem );\n\
\t},\n\
\tchildren: function( elem ) {\n\
\t\treturn jQuery.sibling( elem.firstChild );\n\
\t},\n\
\tcontents: function( elem ) {\n\
\t\treturn jQuery.nodeName( elem, \"iframe\" ) ?\n\
\t\t\telem.contentDocument || elem.contentWindow.document :\n\
\t\t\tjQuery.merge( [], elem.childNodes );\n\
\t}\n\
}, function( name, fn ) {\n\
\tjQuery.fn[ name ] = function( until, selector ) {\n\
\t\tvar ret = jQuery.map( this, fn, until );\n\
\n\
\t\tif ( name.slice( -5 ) !== \"Until\" ) {\n\
\t\t\tselector = until;\n\
\t\t}\n\
\n\
\t\tif ( selector && typeof selector === \"string\" ) {\n\
\t\t\tret = jQuery.filter( selector, ret );\n\
\t\t}\n\
\n\
\t\tif ( this.length > 1 ) {\n\
\t\t\t// Remove duplicates\n\
\t\t\tif ( !guaranteedUnique[ name ] ) {\n\
\t\t\t\tret = jQuery.unique( ret );\n\
\t\t\t}\n\
\n\
\t\t\t// Reverse order for parents* and prev-derivatives\n\
\t\t\tif ( rparentsprev.test( name ) ) {\n\
\t\t\t\tret = ret.reverse();\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn this.pushStack( ret );\n\
\t};\n\
});\n\
var rnotwhite = (/\\S+/g);\n\
\n\
\n\
\n\
// String to Object options format cache\n\
var optionsCache = {};\n\
\n\
// Convert String-formatted options into Object-formatted ones and store in cache\n\
function createOptions( options ) {\n\
\tvar object = optionsCache[ options ] = {};\n\
\tjQuery.each( options.match( rnotwhite ) || [], function( _, flag ) {\n\
\t\tobject[ flag ] = true;\n\
\t});\n\
\treturn object;\n\
}\n\
\n\
/*\n\
 * Create a callback list using the following parameters:\n\
 *\n\
 *\toptions: an optional list of space-separated options that will change how\n\
 *\t\t\tthe callback list behaves or a more traditional option object\n\
 *\n\
 * By default a callback list will act like an event callback list and can be\n\
 * \"fired\" multiple times.\n\
 *\n\
 * Possible options:\n\
 *\n\
 *\tonce:\t\t\twill ensure the callback list can only be fired once (like a Deferred)\n\
 *\n\
 *\tmemory:\t\t\twill keep track of previous values and will call any callback added\n\
 *\t\t\t\t\tafter the list has been fired right away with the latest \"memorized\"\n\
 *\t\t\t\t\tvalues (like a Deferred)\n\
 *\n\
 *\tunique:\t\t\twill ensure a callback can only be added once (no duplicate in the list)\n\
 *\n\
 *\tstopOnFalse:\tinterrupt callings when a callback returns false\n\
 *\n\
 */\n\
jQuery.Callbacks = function( options ) {\n\
\n\
\t// Convert options from String-formatted to Object-formatted if needed\n\
\t// (we check in cache first)\n\
\toptions = typeof options === \"string\" ?\n\
\t\t( optionsCache[ options ] || createOptions( options ) ) :\n\
\t\tjQuery.extend( {}, options );\n\
\n\
\tvar // Flag to know if list is currently firing\n\
\t\tfiring,\n\
\t\t// Last fire value (for non-forgettable lists)\n\
\t\tmemory,\n\
\t\t// Flag to know if list was already fired\n\
\t\tfired,\n\
\t\t// End of the loop when firing\n\
\t\tfiringLength,\n\
\t\t// Index of currently firing callback (modified by remove if needed)\n\
\t\tfiringIndex,\n\
\t\t// First callback to fire (used internally by add and fireWith)\n\
\t\tfiringStart,\n\
\t\t// Actual callback list\n\
\t\tlist = [],\n\
\t\t// Stack of fire calls for repeatable lists\n\
\t\tstack = !options.once && [],\n\
\t\t// Fire callbacks\n\
\t\tfire = function( data ) {\n\
\t\t\tmemory = options.memory && data;\n\
\t\t\tfired = true;\n\
\t\t\tfiringIndex = firingStart || 0;\n\
\t\t\tfiringStart = 0;\n\
\t\t\tfiringLength = list.length;\n\
\t\t\tfiring = true;\n\
\t\t\tfor ( ; list && firingIndex < firingLength; firingIndex++ ) {\n\
\t\t\t\tif ( list[ firingIndex ].apply( data[ 0 ], data[ 1 ] ) === false && options.stopOnFalse ) {\n\
\t\t\t\t\tmemory = false; // To prevent further calls using add\n\
\t\t\t\t\tbreak;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t\tfiring = false;\n\
\t\t\tif ( list ) {\n\
\t\t\t\tif ( stack ) {\n\
\t\t\t\t\tif ( stack.length ) {\n\
\t\t\t\t\t\tfire( stack.shift() );\n\
\t\t\t\t\t}\n\
\t\t\t\t} else if ( memory ) {\n\
\t\t\t\t\tlist = [];\n\
\t\t\t\t} else {\n\
\t\t\t\t\tself.disable();\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t},\n\
\t\t// Actual Callbacks object\n\
\t\tself = {\n\
\t\t\t// Add a callback or a collection of callbacks to the list\n\
\t\t\tadd: function() {\n\
\t\t\t\tif ( list ) {\n\
\t\t\t\t\t// First, we save the current length\n\
\t\t\t\t\tvar start = list.length;\n\
\t\t\t\t\t(function add( args ) {\n\
\t\t\t\t\t\tjQuery.each( args, function( _, arg ) {\n\
\t\t\t\t\t\t\tvar type = jQuery.type( arg );\n\
\t\t\t\t\t\t\tif ( type === \"function\" ) {\n\
\t\t\t\t\t\t\t\tif ( !options.unique || !self.has( arg ) ) {\n\
\t\t\t\t\t\t\t\t\tlist.push( arg );\n\
\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t} else if ( arg && arg.length && type !== \"string\" ) {\n\
\t\t\t\t\t\t\t\t// Inspect recursively\n\
\t\t\t\t\t\t\t\tadd( arg );\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t});\n\
\t\t\t\t\t})( arguments );\n\
\t\t\t\t\t// Do we need to add the callbacks to the\n\
\t\t\t\t\t// current firing batch?\n\
\t\t\t\t\tif ( firing ) {\n\
\t\t\t\t\t\tfiringLength = list.length;\n\
\t\t\t\t\t// With memory, if we're not firing then\n\
\t\t\t\t\t// we should call right away\n\
\t\t\t\t\t} else if ( memory ) {\n\
\t\t\t\t\t\tfiringStart = start;\n\
\t\t\t\t\t\tfire( memory );\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t\treturn this;\n\
\t\t\t},\n\
\t\t\t// Remove a callback from the list\n\
\t\t\tremove: function() {\n\
\t\t\t\tif ( list ) {\n\
\t\t\t\t\tjQuery.each( arguments, function( _, arg ) {\n\
\t\t\t\t\t\tvar index;\n\
\t\t\t\t\t\twhile ( ( index = jQuery.inArray( arg, list, index ) ) > -1 ) {\n\
\t\t\t\t\t\t\tlist.splice( index, 1 );\n\
\t\t\t\t\t\t\t// Handle firing indexes\n\
\t\t\t\t\t\t\tif ( firing ) {\n\
\t\t\t\t\t\t\t\tif ( index <= firingLength ) {\n\
\t\t\t\t\t\t\t\t\tfiringLength--;\n\
\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t\tif ( index <= firingIndex ) {\n\
\t\t\t\t\t\t\t\t\tfiringIndex--;\n\
\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t});\n\
\t\t\t\t}\n\
\t\t\t\treturn this;\n\
\t\t\t},\n\
\t\t\t// Check if a given callback is in the list.\n\
\t\t\t// If no argument is given, return whether or not list has callbacks attached.\n\
\t\t\thas: function( fn ) {\n\
\t\t\t\treturn fn ? jQuery.inArray( fn, list ) > -1 : !!( list && list.length );\n\
\t\t\t},\n\
\t\t\t// Remove all callbacks from the list\n\
\t\t\tempty: function() {\n\
\t\t\t\tlist = [];\n\
\t\t\t\tfiringLength = 0;\n\
\t\t\t\treturn this;\n\
\t\t\t},\n\
\t\t\t// Have the list do nothing anymore\n\
\t\t\tdisable: function() {\n\
\t\t\t\tlist = stack = memory = undefined;\n\
\t\t\t\treturn this;\n\
\t\t\t},\n\
\t\t\t// Is it disabled?\n\
\t\t\tdisabled: function() {\n\
\t\t\t\treturn !list;\n\
\t\t\t},\n\
\t\t\t// Lock the list in its current state\n\
\t\t\tlock: function() {\n\
\t\t\t\tstack = undefined;\n\
\t\t\t\tif ( !memory ) {\n\
\t\t\t\t\tself.disable();\n\
\t\t\t\t}\n\
\t\t\t\treturn this;\n\
\t\t\t},\n\
\t\t\t// Is it locked?\n\
\t\t\tlocked: function() {\n\
\t\t\t\treturn !stack;\n\
\t\t\t},\n\
\t\t\t// Call all callbacks with the given context and arguments\n\
\t\t\tfireWith: function( context, args ) {\n\
\t\t\t\tif ( list && ( !fired || stack ) ) {\n\
\t\t\t\t\targs = args || [];\n\
\t\t\t\t\targs = [ context, args.slice ? args.slice() : args ];\n\
\t\t\t\t\tif ( firing ) {\n\
\t\t\t\t\t\tstack.push( args );\n\
\t\t\t\t\t} else {\n\
\t\t\t\t\t\tfire( args );\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t\treturn this;\n\
\t\t\t},\n\
\t\t\t// Call all the callbacks with the given arguments\n\
\t\t\tfire: function() {\n\
\t\t\t\tself.fireWith( this, arguments );\n\
\t\t\t\treturn this;\n\
\t\t\t},\n\
\t\t\t// To know if the callbacks have already been called at least once\n\
\t\t\tfired: function() {\n\
\t\t\t\treturn !!fired;\n\
\t\t\t}\n\
\t\t};\n\
\n\
\treturn self;\n\
};\n\
\n\
\n\
jQuery.extend({\n\
\n\
\tDeferred: function( func ) {\n\
\t\tvar tuples = [\n\
\t\t\t\t// action, add listener, listener list, final state\n\
\t\t\t\t[ \"resolve\", \"done\", jQuery.Callbacks(\"once memory\"), \"resolved\" ],\n\
\t\t\t\t[ \"reject\", \"fail\", jQuery.Callbacks(\"once memory\"), \"rejected\" ],\n\
\t\t\t\t[ \"notify\", \"progress\", jQuery.Callbacks(\"memory\") ]\n\
\t\t\t],\n\
\t\t\tstate = \"pending\",\n\
\t\t\tpromise = {\n\
\t\t\t\tstate: function() {\n\
\t\t\t\t\treturn state;\n\
\t\t\t\t},\n\
\t\t\t\talways: function() {\n\
\t\t\t\t\tdeferred.done( arguments ).fail( arguments );\n\
\t\t\t\t\treturn this;\n\
\t\t\t\t},\n\
\t\t\t\tthen: function( /* fnDone, fnFail, fnProgress */ ) {\n\
\t\t\t\t\tvar fns = arguments;\n\
\t\t\t\t\treturn jQuery.Deferred(function( newDefer ) {\n\
\t\t\t\t\t\tjQuery.each( tuples, function( i, tuple ) {\n\
\t\t\t\t\t\t\tvar fn = jQuery.isFunction( fns[ i ] ) && fns[ i ];\n\
\t\t\t\t\t\t\t// deferred[ done | fail | progress ] for forwarding actions to newDefer\n\
\t\t\t\t\t\t\tdeferred[ tuple[1] ](function() {\n\
\t\t\t\t\t\t\t\tvar returned = fn && fn.apply( this, arguments );\n\
\t\t\t\t\t\t\t\tif ( returned && jQuery.isFunction( returned.promise ) ) {\n\
\t\t\t\t\t\t\t\t\treturned.promise()\n\
\t\t\t\t\t\t\t\t\t\t.done( newDefer.resolve )\n\
\t\t\t\t\t\t\t\t\t\t.fail( newDefer.reject )\n\
\t\t\t\t\t\t\t\t\t\t.progress( newDefer.notify );\n\
\t\t\t\t\t\t\t\t} else {\n\
\t\t\t\t\t\t\t\t\tnewDefer[ tuple[ 0 ] + \"With\" ]( this === promise ? newDefer.promise() : this, fn ? [ returned ] : arguments );\n\
\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t});\n\
\t\t\t\t\t\t});\n\
\t\t\t\t\t\tfns = null;\n\
\t\t\t\t\t}).promise();\n\
\t\t\t\t},\n\
\t\t\t\t// Get a promise for this deferred\n\
\t\t\t\t// If obj is provided, the promise aspect is added to the object\n\
\t\t\t\tpromise: function( obj ) {\n\
\t\t\t\t\treturn obj != null ? jQuery.extend( obj, promise ) : promise;\n\
\t\t\t\t}\n\
\t\t\t},\n\
\t\t\tdeferred = {};\n\
\n\
\t\t// Keep pipe for back-compat\n\
\t\tpromise.pipe = promise.then;\n\
\n\
\t\t// Add list-specific methods\n\
\t\tjQuery.each( tuples, function( i, tuple ) {\n\
\t\t\tvar list = tuple[ 2 ],\n\
\t\t\t\tstateString = tuple[ 3 ];\n\
\n\
\t\t\t// promise[ done | fail | progress ] = list.add\n\
\t\t\tpromise[ tuple[1] ] = list.add;\n\
\n\
\t\t\t// Handle state\n\
\t\t\tif ( stateString ) {\n\
\t\t\t\tlist.add(function() {\n\
\t\t\t\t\t// state = [ resolved | rejected ]\n\
\t\t\t\t\tstate = stateString;\n\
\n\
\t\t\t\t// [ reject_list | resolve_list ].disable; progress_list.lock\n\
\t\t\t\t}, tuples[ i ^ 1 ][ 2 ].disable, tuples[ 2 ][ 2 ].lock );\n\
\t\t\t}\n\
\n\
\t\t\t// deferred[ resolve | reject | notify ]\n\
\t\t\tdeferred[ tuple[0] ] = function() {\n\
\t\t\t\tdeferred[ tuple[0] + \"With\" ]( this === deferred ? promise : this, arguments );\n\
\t\t\t\treturn this;\n\
\t\t\t};\n\
\t\t\tdeferred[ tuple[0] + \"With\" ] = list.fireWith;\n\
\t\t});\n\
\n\
\t\t// Make the deferred a promise\n\
\t\tpromise.promise( deferred );\n\
\n\
\t\t// Call given func if any\n\
\t\tif ( func ) {\n\
\t\t\tfunc.call( deferred, deferred );\n\
\t\t}\n\
\n\
\t\t// All done!\n\
\t\treturn deferred;\n\
\t},\n\
\n\
\t// Deferred helper\n\
\twhen: function( subordinate /* , ..., subordinateN */ ) {\n\
\t\tvar i = 0,\n\
\t\t\tresolveValues = slice.call( arguments ),\n\
\t\t\tlength = resolveValues.length,\n\
\n\
\t\t\t// the count of uncompleted subordinates\n\
\t\t\tremaining = length !== 1 || ( subordinate && jQuery.isFunction( subordinate.promise ) ) ? length : 0,\n\
\n\
\t\t\t// the master Deferred. If resolveValues consist of only a single Deferred, just use that.\n\
\t\t\tdeferred = remaining === 1 ? subordinate : jQuery.Deferred(),\n\
\n\
\t\t\t// Update function for both resolve and progress values\n\
\t\t\tupdateFunc = function( i, contexts, values ) {\n\
\t\t\t\treturn function( value ) {\n\
\t\t\t\t\tcontexts[ i ] = this;\n\
\t\t\t\t\tvalues[ i ] = arguments.length > 1 ? slice.call( arguments ) : value;\n\
\t\t\t\t\tif ( values === progressValues ) {\n\
\t\t\t\t\t\tdeferred.notifyWith( contexts, values );\n\
\n\
\t\t\t\t\t} else if ( !(--remaining) ) {\n\
\t\t\t\t\t\tdeferred.resolveWith( contexts, values );\n\
\t\t\t\t\t}\n\
\t\t\t\t};\n\
\t\t\t},\n\
\n\
\t\t\tprogressValues, progressContexts, resolveContexts;\n\
\n\
\t\t// add listeners to Deferred subordinates; treat others as resolved\n\
\t\tif ( length > 1 ) {\n\
\t\t\tprogressValues = new Array( length );\n\
\t\t\tprogressContexts = new Array( length );\n\
\t\t\tresolveContexts = new Array( length );\n\
\t\t\tfor ( ; i < length; i++ ) {\n\
\t\t\t\tif ( resolveValues[ i ] && jQuery.isFunction( resolveValues[ i ].promise ) ) {\n\
\t\t\t\t\tresolveValues[ i ].promise()\n\
\t\t\t\t\t\t.done( updateFunc( i, resolveContexts, resolveValues ) )\n\
\t\t\t\t\t\t.fail( deferred.reject )\n\
\t\t\t\t\t\t.progress( updateFunc( i, progressContexts, progressValues ) );\n\
\t\t\t\t} else {\n\
\t\t\t\t\t--remaining;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// if we're not waiting on anything, resolve the master\n\
\t\tif ( !remaining ) {\n\
\t\t\tdeferred.resolveWith( resolveContexts, resolveValues );\n\
\t\t}\n\
\n\
\t\treturn deferred.promise();\n\
\t}\n\
});\n\
\n\
\n\
// The deferred used on DOM ready\n\
var readyList;\n\
\n\
jQuery.fn.ready = function( fn ) {\n\
\t// Add the callback\n\
\tjQuery.ready.promise().done( fn );\n\
\n\
\treturn this;\n\
};\n\
\n\
jQuery.extend({\n\
\t// Is the DOM ready to be used? Set to true once it occurs.\n\
\tisReady: false,\n\
\n\
\t// A counter to track how many items to wait for before\n\
\t// the ready event fires. See #6781\n\
\treadyWait: 1,\n\
\n\
\t// Hold (or release) the ready event\n\
\tholdReady: function( hold ) {\n\
\t\tif ( hold ) {\n\
\t\t\tjQuery.readyWait++;\n\
\t\t} else {\n\
\t\t\tjQuery.ready( true );\n\
\t\t}\n\
\t},\n\
\n\
\t// Handle when the DOM is ready\n\
\tready: function( wait ) {\n\
\n\
\t\t// Abort if there are pending holds or we're already ready\n\
\t\tif ( wait === true ? --jQuery.readyWait : jQuery.isReady ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\t// Make sure body exists, at least, in case IE gets a little overzealous (ticket #5443).\n\
\t\tif ( !document.body ) {\n\
\t\t\treturn setTimeout( jQuery.ready );\n\
\t\t}\n\
\n\
\t\t// Remember that the DOM is ready\n\
\t\tjQuery.isReady = true;\n\
\n\
\t\t// If a normal DOM Ready event fired, decrement, and wait if need be\n\
\t\tif ( wait !== true && --jQuery.readyWait > 0 ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\t// If there are functions bound, to execute\n\
\t\treadyList.resolveWith( document, [ jQuery ] );\n\
\n\
\t\t// Trigger any bound ready events\n\
\t\tif ( jQuery.fn.triggerHandler ) {\n\
\t\t\tjQuery( document ).triggerHandler( \"ready\" );\n\
\t\t\tjQuery( document ).off( \"ready\" );\n\
\t\t}\n\
\t}\n\
});\n\
\n\
/**\n\
 * Clean-up method for dom ready events\n\
 */\n\
function detach() {\n\
\tif ( document.addEventListener ) {\n\
\t\tdocument.removeEventListener( \"DOMContentLoaded\", completed, false );\n\
\t\twindow.removeEventListener( \"load\", completed, false );\n\
\n\
\t} else {\n\
\t\tdocument.detachEvent( \"onreadystatechange\", completed );\n\
\t\twindow.detachEvent( \"onload\", completed );\n\
\t}\n\
}\n\
\n\
/**\n\
 * The ready event handler and self cleanup method\n\
 */\n\
function completed() {\n\
\t// readyState === \"complete\" is good enough for us to call the dom ready in oldIE\n\
\tif ( document.addEventListener || event.type === \"load\" || document.readyState === \"complete\" ) {\n\
\t\tdetach();\n\
\t\tjQuery.ready();\n\
\t}\n\
}\n\
\n\
jQuery.ready.promise = function( obj ) {\n\
\tif ( !readyList ) {\n\
\n\
\t\treadyList = jQuery.Deferred();\n\
\n\
\t\t// Catch cases where $(document).ready() is called after the browser event has already occurred.\n\
\t\t// we once tried to use readyState \"interactive\" here, but it caused issues like the one\n\
\t\t// discovered by ChrisS here: http://bugs.jquery.com/ticket/12282#comment:15\n\
\t\tif ( document.readyState === \"complete\" ) {\n\
\t\t\t// Handle it asynchronously to allow scripts the opportunity to delay ready\n\
\t\t\tsetTimeout( jQuery.ready );\n\
\n\
\t\t// Standards-based browsers support DOMContentLoaded\n\
\t\t} else if ( document.addEventListener ) {\n\
\t\t\t// Use the handy event callback\n\
\t\t\tdocument.addEventListener( \"DOMContentLoaded\", completed, false );\n\
\n\
\t\t\t// A fallback to window.onload, that will always work\n\
\t\t\twindow.addEventListener( \"load\", completed, false );\n\
\n\
\t\t// If IE event model is used\n\
\t\t} else {\n\
\t\t\t// Ensure firing before onload, maybe late but safe also for iframes\n\
\t\t\tdocument.attachEvent( \"onreadystatechange\", completed );\n\
\n\
\t\t\t// A fallback to window.onload, that will always work\n\
\t\t\twindow.attachEvent( \"onload\", completed );\n\
\n\
\t\t\t// If IE and not a frame\n\
\t\t\t// continually check to see if the document is ready\n\
\t\t\tvar top = false;\n\
\n\
\t\t\ttry {\n\
\t\t\t\ttop = window.frameElement == null && document.documentElement;\n\
\t\t\t} catch(e) {}\n\
\n\
\t\t\tif ( top && top.doScroll ) {\n\
\t\t\t\t(function doScrollCheck() {\n\
\t\t\t\t\tif ( !jQuery.isReady ) {\n\
\n\
\t\t\t\t\t\ttry {\n\
\t\t\t\t\t\t\t// Use the trick by Diego Perini\n\
\t\t\t\t\t\t\t// http://javascript.nwbox.com/IEContentLoaded/\n\
\t\t\t\t\t\t\ttop.doScroll(\"left\");\n\
\t\t\t\t\t\t} catch(e) {\n\
\t\t\t\t\t\t\treturn setTimeout( doScrollCheck, 50 );\n\
\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t\t// detach all dom ready events\n\
\t\t\t\t\t\tdetach();\n\
\n\
\t\t\t\t\t\t// and execute any waiting functions\n\
\t\t\t\t\t\tjQuery.ready();\n\
\t\t\t\t\t}\n\
\t\t\t\t})();\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\treturn readyList.promise( obj );\n\
};\n\
\n\
\n\
var strundefined = typeof undefined;\n\
\n\
\n\
\n\
// Support: IE<9\n\
// Iteration over object's inherited properties before its own\n\
var i;\n\
for ( i in jQuery( support ) ) {\n\
\tbreak;\n\
}\n\
support.ownLast = i !== \"0\";\n\
\n\
// Note: most support tests are defined in their respective modules.\n\
// false until the test is run\n\
support.inlineBlockNeedsLayout = false;\n\
\n\
// Execute ASAP in case we need to set body.style.zoom\n\
jQuery(function() {\n\
\t// Minified: var a,b,c,d\n\
\tvar val, div, body, container;\n\
\n\
\tbody = document.getElementsByTagName( \"body\" )[ 0 ];\n\
\tif ( !body || !body.style ) {\n\
\t\t// Return for frameset docs that don't have a body\n\
\t\treturn;\n\
\t}\n\
\n\
\t// Setup\n\
\tdiv = document.createElement( \"div\" );\n\
\tcontainer = document.createElement( \"div\" );\n\
\tcontainer.style.cssText = \"position:absolute;border:0;width:0;height:0;top:0;left:-9999px\";\n\
\tbody.appendChild( container ).appendChild( div );\n\
\n\
\tif ( typeof div.style.zoom !== strundefined ) {\n\
\t\t// Support: IE<8\n\
\t\t// Check if natively block-level elements act like inline-block\n\
\t\t// elements when setting their display to 'inline' and giving\n\
\t\t// them layout\n\
\t\tdiv.style.cssText = \"display:inline;margin:0;border:0;padding:1px;width:1px;zoom:1\";\n\
\n\
\t\tsupport.inlineBlockNeedsLayout = val = div.offsetWidth === 3;\n\
\t\tif ( val ) {\n\
\t\t\t// Prevent IE 6 from affecting layout for positioned elements #11048\n\
\t\t\t// Prevent IE from shrinking the body in IE 7 mode #12869\n\
\t\t\t// Support: IE<8\n\
\t\t\tbody.style.zoom = 1;\n\
\t\t}\n\
\t}\n\
\n\
\tbody.removeChild( container );\n\
});\n\
\n\
\n\
\n\
\n\
(function() {\n\
\tvar div = document.createElement( \"div\" );\n\
\n\
\t// Execute the test only if not already executed in another module.\n\
\tif (support.deleteExpando == null) {\n\
\t\t// Support: IE<9\n\
\t\tsupport.deleteExpando = true;\n\
\t\ttry {\n\
\t\t\tdelete div.test;\n\
\t\t} catch( e ) {\n\
\t\t\tsupport.deleteExpando = false;\n\
\t\t}\n\
\t}\n\
\n\
\t// Null elements to avoid leaks in IE.\n\
\tdiv = null;\n\
})();\n\
\n\
\n\
/**\n\
 * Determines whether an object can have data\n\
 */\n\
jQuery.acceptData = function( elem ) {\n\
\tvar noData = jQuery.noData[ (elem.nodeName + \" \").toLowerCase() ],\n\
\t\tnodeType = +elem.nodeType || 1;\n\
\n\
\t// Do not set data on non-element DOM nodes because it will not be cleared (#8335).\n\
\treturn nodeType !== 1 && nodeType !== 9 ?\n\
\t\tfalse :\n\
\n\
\t\t// Nodes accept data unless otherwise specified; rejection can be conditional\n\
\t\t!noData || noData !== true && elem.getAttribute(\"classid\") === noData;\n\
};\n\
\n\
\n\
var rbrace = /^(?:\\{[\\w\\W]*\\}|\\[[\\w\\W]*\\])$/,\n\
\trmultiDash = /([A-Z])/g;\n\
\n\
function dataAttr( elem, key, data ) {\n\
\t// If nothing was found internally, try to fetch any\n\
\t// data from the HTML5 data-* attribute\n\
\tif ( data === undefined && elem.nodeType === 1 ) {\n\
\n\
\t\tvar name = \"data-\" + key.replace( rmultiDash, \"-$1\" ).toLowerCase();\n\
\n\
\t\tdata = elem.getAttribute( name );\n\
\n\
\t\tif ( typeof data === \"string\" ) {\n\
\t\t\ttry {\n\
\t\t\t\tdata = data === \"true\" ? true :\n\
\t\t\t\t\tdata === \"false\" ? false :\n\
\t\t\t\t\tdata === \"null\" ? null :\n\
\t\t\t\t\t// Only convert to a number if it doesn't change the string\n\
\t\t\t\t\t+data + \"\" === data ? +data :\n\
\t\t\t\t\trbrace.test( data ) ? jQuery.parseJSON( data ) :\n\
\t\t\t\t\tdata;\n\
\t\t\t} catch( e ) {}\n\
\n\
\t\t\t// Make sure we set the data so it isn't changed later\n\
\t\t\tjQuery.data( elem, key, data );\n\
\n\
\t\t} else {\n\
\t\t\tdata = undefined;\n\
\t\t}\n\
\t}\n\
\n\
\treturn data;\n\
}\n\
\n\
// checks a cache object for emptiness\n\
function isEmptyDataObject( obj ) {\n\
\tvar name;\n\
\tfor ( name in obj ) {\n\
\n\
\t\t// if the public data object is empty, the private is still empty\n\
\t\tif ( name === \"data\" && jQuery.isEmptyObject( obj[name] ) ) {\n\
\t\t\tcontinue;\n\
\t\t}\n\
\t\tif ( name !== \"toJSON\" ) {\n\
\t\t\treturn false;\n\
\t\t}\n\
\t}\n\
\n\
\treturn true;\n\
}\n\
\n\
function internalData( elem, name, data, pvt /* Internal Use Only */ ) {\n\
\tif ( !jQuery.acceptData( elem ) ) {\n\
\t\treturn;\n\
\t}\n\
\n\
\tvar ret, thisCache,\n\
\t\tinternalKey = jQuery.expando,\n\
\n\
\t\t// We have to handle DOM nodes and JS objects differently because IE6-7\n\
\t\t// can't GC object references properly across the DOM-JS boundary\n\
\t\tisNode = elem.nodeType,\n\
\n\
\t\t// Only DOM nodes need the global jQuery cache; JS object data is\n\
\t\t// attached directly to the object so GC can occur automatically\n\
\t\tcache = isNode ? jQuery.cache : elem,\n\
\n\
\t\t// Only defining an ID for JS objects if its cache already exists allows\n\
\t\t// the code to shortcut on the same path as a DOM node with no cache\n\
\t\tid = isNode ? elem[ internalKey ] : elem[ internalKey ] && internalKey;\n\
\n\
\t// Avoid doing any more work than we need to when trying to get data on an\n\
\t// object that has no data at all\n\
\tif ( (!id || !cache[id] || (!pvt && !cache[id].data)) && data === undefined && typeof name === \"string\" ) {\n\
\t\treturn;\n\
\t}\n\
\n\
\tif ( !id ) {\n\
\t\t// Only DOM nodes need a new unique ID for each element since their data\n\
\t\t// ends up in the global cache\n\
\t\tif ( isNode ) {\n\
\t\t\tid = elem[ internalKey ] = deletedIds.pop() || jQuery.guid++;\n\
\t\t} else {\n\
\t\t\tid = internalKey;\n\
\t\t}\n\
\t}\n\
\n\
\tif ( !cache[ id ] ) {\n\
\t\t// Avoid exposing jQuery metadata on plain JS objects when the object\n\
\t\t// is serialized using JSON.stringify\n\
\t\tcache[ id ] = isNode ? {} : { toJSON: jQuery.noop };\n\
\t}\n\
\n\
\t// An object can be passed to jQuery.data instead of a key/value pair; this gets\n\
\t// shallow copied over onto the existing cache\n\
\tif ( typeof name === \"object\" || typeof name === \"function\" ) {\n\
\t\tif ( pvt ) {\n\
\t\t\tcache[ id ] = jQuery.extend( cache[ id ], name );\n\
\t\t} else {\n\
\t\t\tcache[ id ].data = jQuery.extend( cache[ id ].data, name );\n\
\t\t}\n\
\t}\n\
\n\
\tthisCache = cache[ id ];\n\
\n\
\t// jQuery data() is stored in a separate object inside the object's internal data\n\
\t// cache in order to avoid key collisions between internal data and user-defined\n\
\t// data.\n\
\tif ( !pvt ) {\n\
\t\tif ( !thisCache.data ) {\n\
\t\t\tthisCache.data = {};\n\
\t\t}\n\
\n\
\t\tthisCache = thisCache.data;\n\
\t}\n\
\n\
\tif ( data !== undefined ) {\n\
\t\tthisCache[ jQuery.camelCase( name ) ] = data;\n\
\t}\n\
\n\
\t// Check for both converted-to-camel and non-converted data property names\n\
\t// If a data property was specified\n\
\tif ( typeof name === \"string\" ) {\n\
\n\
\t\t// First Try to find as-is property data\n\
\t\tret = thisCache[ name ];\n\
\n\
\t\t// Test for null|undefined property data\n\
\t\tif ( ret == null ) {\n\
\n\
\t\t\t// Try to find the camelCased property\n\
\t\t\tret = thisCache[ jQuery.camelCase( name ) ];\n\
\t\t}\n\
\t} else {\n\
\t\tret = thisCache;\n\
\t}\n\
\n\
\treturn ret;\n\
}\n\
\n\
function internalRemoveData( elem, name, pvt ) {\n\
\tif ( !jQuery.acceptData( elem ) ) {\n\
\t\treturn;\n\
\t}\n\
\n\
\tvar thisCache, i,\n\
\t\tisNode = elem.nodeType,\n\
\n\
\t\t// See jQuery.data for more information\n\
\t\tcache = isNode ? jQuery.cache : elem,\n\
\t\tid = isNode ? elem[ jQuery.expando ] : jQuery.expando;\n\
\n\
\t// If there is already no cache entry for this object, there is no\n\
\t// purpose in continuing\n\
\tif ( !cache[ id ] ) {\n\
\t\treturn;\n\
\t}\n\
\n\
\tif ( name ) {\n\
\n\
\t\tthisCache = pvt ? cache[ id ] : cache[ id ].data;\n\
\n\
\t\tif ( thisCache ) {\n\
\n\
\t\t\t// Support array or space separated string names for data keys\n\
\t\t\tif ( !jQuery.isArray( name ) ) {\n\
\n\
\t\t\t\t// try the string as a key before any manipulation\n\
\t\t\t\tif ( name in thisCache ) {\n\
\t\t\t\t\tname = [ name ];\n\
\t\t\t\t} else {\n\
\n\
\t\t\t\t\t// split the camel cased version by spaces unless a key with the spaces exists\n\
\t\t\t\t\tname = jQuery.camelCase( name );\n\
\t\t\t\t\tif ( name in thisCache ) {\n\
\t\t\t\t\t\tname = [ name ];\n\
\t\t\t\t\t} else {\n\
\t\t\t\t\t\tname = name.split(\" \");\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t} else {\n\
\t\t\t\t// If \"name\" is an array of keys...\n\
\t\t\t\t// When data is initially created, via (\"key\", \"val\") signature,\n\
\t\t\t\t// keys will be converted to camelCase.\n\
\t\t\t\t// Since there is no way to tell _how_ a key was added, remove\n\
\t\t\t\t// both plain key and camelCase key. #12786\n\
\t\t\t\t// This will only penalize the array argument path.\n\
\t\t\t\tname = name.concat( jQuery.map( name, jQuery.camelCase ) );\n\
\t\t\t}\n\
\n\
\t\t\ti = name.length;\n\
\t\t\twhile ( i-- ) {\n\
\t\t\t\tdelete thisCache[ name[i] ];\n\
\t\t\t}\n\
\n\
\t\t\t// If there is no data left in the cache, we want to continue\n\
\t\t\t// and let the cache object itself get destroyed\n\
\t\t\tif ( pvt ? !isEmptyDataObject(thisCache) : !jQuery.isEmptyObject(thisCache) ) {\n\
\t\t\t\treturn;\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\t// See jQuery.data for more information\n\
\tif ( !pvt ) {\n\
\t\tdelete cache[ id ].data;\n\
\n\
\t\t// Don't destroy the parent cache unless the internal data object\n\
\t\t// had been the only thing left in it\n\
\t\tif ( !isEmptyDataObject( cache[ id ] ) ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\t}\n\
\n\
\t// Destroy the cache\n\
\tif ( isNode ) {\n\
\t\tjQuery.cleanData( [ elem ], true );\n\
\n\
\t// Use delete when supported for expandos or `cache` is not a window per isWindow (#10080)\n\
\t/* jshint eqeqeq: false */\n\
\t} else if ( support.deleteExpando || cache != cache.window ) {\n\
\t\t/* jshint eqeqeq: true */\n\
\t\tdelete cache[ id ];\n\
\n\
\t// When all else fails, null\n\
\t} else {\n\
\t\tcache[ id ] = null;\n\
\t}\n\
}\n\
\n\
jQuery.extend({\n\
\tcache: {},\n\
\n\
\t// The following elements (space-suffixed to avoid Object.prototype collisions)\n\
\t// throw uncatchable exceptions if you attempt to set expando properties\n\
\tnoData: {\n\
\t\t\"applet \": true,\n\
\t\t\"embed \": true,\n\
\t\t// ...but Flash objects (which have this classid) *can* handle expandos\n\
\t\t\"object \": \"clsid:D27CDB6E-AE6D-11cf-96B8-444553540000\"\n\
\t},\n\
\n\
\thasData: function( elem ) {\n\
\t\telem = elem.nodeType ? jQuery.cache[ elem[jQuery.expando] ] : elem[ jQuery.expando ];\n\
\t\treturn !!elem && !isEmptyDataObject( elem );\n\
\t},\n\
\n\
\tdata: function( elem, name, data ) {\n\
\t\treturn internalData( elem, name, data );\n\
\t},\n\
\n\
\tremoveData: function( elem, name ) {\n\
\t\treturn internalRemoveData( elem, name );\n\
\t},\n\
\n\
\t// For internal use only.\n\
\t_data: function( elem, name, data ) {\n\
\t\treturn internalData( elem, name, data, true );\n\
\t},\n\
\n\
\t_removeData: function( elem, name ) {\n\
\t\treturn internalRemoveData( elem, name, true );\n\
\t}\n\
});\n\
\n\
jQuery.fn.extend({\n\
\tdata: function( key, value ) {\n\
\t\tvar i, name, data,\n\
\t\t\telem = this[0],\n\
\t\t\tattrs = elem && elem.attributes;\n\
\n\
\t\t// Special expections of .data basically thwart jQuery.access,\n\
\t\t// so implement the relevant behavior ourselves\n\
\n\
\t\t// Gets all values\n\
\t\tif ( key === undefined ) {\n\
\t\t\tif ( this.length ) {\n\
\t\t\t\tdata = jQuery.data( elem );\n\
\n\
\t\t\t\tif ( elem.nodeType === 1 && !jQuery._data( elem, \"parsedAttrs\" ) ) {\n\
\t\t\t\t\ti = attrs.length;\n\
\t\t\t\t\twhile ( i-- ) {\n\
\n\
\t\t\t\t\t\t// Support: IE11+\n\
\t\t\t\t\t\t// The attrs elements can be null (#14894)\n\
\t\t\t\t\t\tif ( attrs[ i ] ) {\n\
\t\t\t\t\t\t\tname = attrs[ i ].name;\n\
\t\t\t\t\t\t\tif ( name.indexOf( \"data-\" ) === 0 ) {\n\
\t\t\t\t\t\t\t\tname = jQuery.camelCase( name.slice(5) );\n\
\t\t\t\t\t\t\t\tdataAttr( elem, name, data[ name ] );\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t\tjQuery._data( elem, \"parsedAttrs\", true );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t\treturn data;\n\
\t\t}\n\
\n\
\t\t// Sets multiple values\n\
\t\tif ( typeof key === \"object\" ) {\n\
\t\t\treturn this.each(function() {\n\
\t\t\t\tjQuery.data( this, key );\n\
\t\t\t});\n\
\t\t}\n\
\n\
\t\treturn arguments.length > 1 ?\n\
\n\
\t\t\t// Sets one value\n\
\t\t\tthis.each(function() {\n\
\t\t\t\tjQuery.data( this, key, value );\n\
\t\t\t}) :\n\
\n\
\t\t\t// Gets one value\n\
\t\t\t// Try to fetch any internally stored data first\n\
\t\t\telem ? dataAttr( elem, key, jQuery.data( elem, key ) ) : undefined;\n\
\t},\n\
\n\
\tremoveData: function( key ) {\n\
\t\treturn this.each(function() {\n\
\t\t\tjQuery.removeData( this, key );\n\
\t\t});\n\
\t}\n\
});\n\
\n\
\n\
jQuery.extend({\n\
\tqueue: function( elem, type, data ) {\n\
\t\tvar queue;\n\
\n\
\t\tif ( elem ) {\n\
\t\t\ttype = ( type || \"fx\" ) + \"queue\";\n\
\t\t\tqueue = jQuery._data( elem, type );\n\
\n\
\t\t\t// Speed up dequeue by getting out quickly if this is just a lookup\n\
\t\t\tif ( data ) {\n\
\t\t\t\tif ( !queue || jQuery.isArray(data) ) {\n\
\t\t\t\t\tqueue = jQuery._data( elem, type, jQuery.makeArray(data) );\n\
\t\t\t\t} else {\n\
\t\t\t\t\tqueue.push( data );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t\treturn queue || [];\n\
\t\t}\n\
\t},\n\
\n\
\tdequeue: function( elem, type ) {\n\
\t\ttype = type || \"fx\";\n\
\n\
\t\tvar queue = jQuery.queue( elem, type ),\n\
\t\t\tstartLength = queue.length,\n\
\t\t\tfn = queue.shift(),\n\
\t\t\thooks = jQuery._queueHooks( elem, type ),\n\
\t\t\tnext = function() {\n\
\t\t\t\tjQuery.dequeue( elem, type );\n\
\t\t\t};\n\
\n\
\t\t// If the fx queue is dequeued, always remove the progress sentinel\n\
\t\tif ( fn === \"inprogress\" ) {\n\
\t\t\tfn = queue.shift();\n\
\t\t\tstartLength--;\n\
\t\t}\n\
\n\
\t\tif ( fn ) {\n\
\n\
\t\t\t// Add a progress sentinel to prevent the fx queue from being\n\
\t\t\t// automatically dequeued\n\
\t\t\tif ( type === \"fx\" ) {\n\
\t\t\t\tqueue.unshift( \"inprogress\" );\n\
\t\t\t}\n\
\n\
\t\t\t// clear up the last queue stop function\n\
\t\t\tdelete hooks.stop;\n\
\t\t\tfn.call( elem, next, hooks );\n\
\t\t}\n\
\n\
\t\tif ( !startLength && hooks ) {\n\
\t\t\thooks.empty.fire();\n\
\t\t}\n\
\t},\n\
\n\
\t// not intended for public consumption - generates a queueHooks object, or returns the current one\n\
\t_queueHooks: function( elem, type ) {\n\
\t\tvar key = type + \"queueHooks\";\n\
\t\treturn jQuery._data( elem, key ) || jQuery._data( elem, key, {\n\
\t\t\tempty: jQuery.Callbacks(\"once memory\").add(function() {\n\
\t\t\t\tjQuery._removeData( elem, type + \"queue\" );\n\
\t\t\t\tjQuery._removeData( elem, key );\n\
\t\t\t})\n\
\t\t});\n\
\t}\n\
});\n\
\n\
jQuery.fn.extend({\n\
\tqueue: function( type, data ) {\n\
\t\tvar setter = 2;\n\
\n\
\t\tif ( typeof type !== \"string\" ) {\n\
\t\t\tdata = type;\n\
\t\t\ttype = \"fx\";\n\
\t\t\tsetter--;\n\
\t\t}\n\
\n\
\t\tif ( arguments.length < setter ) {\n\
\t\t\treturn jQuery.queue( this[0], type );\n\
\t\t}\n\
\n\
\t\treturn data === undefined ?\n\
\t\t\tthis :\n\
\t\t\tthis.each(function() {\n\
\t\t\t\tvar queue = jQuery.queue( this, type, data );\n\
\n\
\t\t\t\t// ensure a hooks for this queue\n\
\t\t\t\tjQuery._queueHooks( this, type );\n\
\n\
\t\t\t\tif ( type === \"fx\" && queue[0] !== \"inprogress\" ) {\n\
\t\t\t\t\tjQuery.dequeue( this, type );\n\
\t\t\t\t}\n\
\t\t\t});\n\
\t},\n\
\tdequeue: function( type ) {\n\
\t\treturn this.each(function() {\n\
\t\t\tjQuery.dequeue( this, type );\n\
\t\t});\n\
\t},\n\
\tclearQueue: function( type ) {\n\
\t\treturn this.queue( type || \"fx\", [] );\n\
\t},\n\
\t// Get a promise resolved when queues of a certain type\n\
\t// are emptied (fx is the type by default)\n\
\tpromise: function( type, obj ) {\n\
\t\tvar tmp,\n\
\t\t\tcount = 1,\n\
\t\t\tdefer = jQuery.Deferred(),\n\
\t\t\telements = this,\n\
\t\t\ti = this.length,\n\
\t\t\tresolve = function() {\n\
\t\t\t\tif ( !( --count ) ) {\n\
\t\t\t\t\tdefer.resolveWith( elements, [ elements ] );\n\
\t\t\t\t}\n\
\t\t\t};\n\
\n\
\t\tif ( typeof type !== \"string\" ) {\n\
\t\t\tobj = type;\n\
\t\t\ttype = undefined;\n\
\t\t}\n\
\t\ttype = type || \"fx\";\n\
\n\
\t\twhile ( i-- ) {\n\
\t\t\ttmp = jQuery._data( elements[ i ], type + \"queueHooks\" );\n\
\t\t\tif ( tmp && tmp.empty ) {\n\
\t\t\t\tcount++;\n\
\t\t\t\ttmp.empty.add( resolve );\n\
\t\t\t}\n\
\t\t}\n\
\t\tresolve();\n\
\t\treturn defer.promise( obj );\n\
\t}\n\
});\n\
var pnum = (/[+-]?(?:\\d*\\.|)\\d+(?:[eE][+-]?\\d+|)/).source;\n\
\n\
var cssExpand = [ \"Top\", \"Right\", \"Bottom\", \"Left\" ];\n\
\n\
var isHidden = function( elem, el ) {\n\
\t\t// isHidden might be called from jQuery#filter function;\n\
\t\t// in that case, element will be second argument\n\
\t\telem = el || elem;\n\
\t\treturn jQuery.css( elem, \"display\" ) === \"none\" || !jQuery.contains( elem.ownerDocument, elem );\n\
\t};\n\
\n\
\n\
\n\
// Multifunctional method to get and set values of a collection\n\
// The value/s can optionally be executed if it's a function\n\
var access = jQuery.access = function( elems, fn, key, value, chainable, emptyGet, raw ) {\n\
\tvar i = 0,\n\
\t\tlength = elems.length,\n\
\t\tbulk = key == null;\n\
\n\
\t// Sets many values\n\
\tif ( jQuery.type( key ) === \"object\" ) {\n\
\t\tchainable = true;\n\
\t\tfor ( i in key ) {\n\
\t\t\tjQuery.access( elems, fn, i, key[i], true, emptyGet, raw );\n\
\t\t}\n\
\n\
\t// Sets one value\n\
\t} else if ( value !== undefined ) {\n\
\t\tchainable = true;\n\
\n\
\t\tif ( !jQuery.isFunction( value ) ) {\n\
\t\t\traw = true;\n\
\t\t}\n\
\n\
\t\tif ( bulk ) {\n\
\t\t\t// Bulk operations run against the entire set\n\
\t\t\tif ( raw ) {\n\
\t\t\t\tfn.call( elems, value );\n\
\t\t\t\tfn = null;\n\
\n\
\t\t\t// ...except when executing function values\n\
\t\t\t} else {\n\
\t\t\t\tbulk = fn;\n\
\t\t\t\tfn = function( elem, key, value ) {\n\
\t\t\t\t\treturn bulk.call( jQuery( elem ), value );\n\
\t\t\t\t};\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\tif ( fn ) {\n\
\t\t\tfor ( ; i < length; i++ ) {\n\
\t\t\t\tfn( elems[i], key, raw ? value : value.call( elems[i], i, fn( elems[i], key ) ) );\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\treturn chainable ?\n\
\t\telems :\n\
\n\
\t\t// Gets\n\
\t\tbulk ?\n\
\t\t\tfn.call( elems ) :\n\
\t\t\tlength ? fn( elems[0], key ) : emptyGet;\n\
};\n\
var rcheckableType = (/^(?:checkbox|radio)$/i);\n\
\n\
\n\
\n\
(function() {\n\
\t// Minified: var a,b,c\n\
\tvar input = document.createElement( \"input\" ),\n\
\t\tdiv = document.createElement( \"div\" ),\n\
\t\tfragment = document.createDocumentFragment();\n\
\n\
\t// Setup\n\
\tdiv.innerHTML = \"  <link/><table></table><a href='/a'>a</a><input type='checkbox'/>\";\n\
\n\
\t// IE strips leading whitespace when .innerHTML is used\n\
\tsupport.leadingWhitespace = div.firstChild.nodeType === 3;\n\
\n\
\t// Make sure that tbody elements aren't automatically inserted\n\
\t// IE will insert them into empty tables\n\
\tsupport.tbody = !div.getElementsByTagName( \"tbody\" ).length;\n\
\n\
\t// Make sure that link elements get serialized correctly by innerHTML\n\
\t// This requires a wrapper element in IE\n\
\tsupport.htmlSerialize = !!div.getElementsByTagName( \"link\" ).length;\n\
\n\
\t// Makes sure cloning an html5 element does not cause problems\n\
\t// Where outerHTML is undefined, this still works\n\
\tsupport.html5Clone =\n\
\t\tdocument.createElement( \"nav\" ).cloneNode( true ).outerHTML !== \"<:nav></:nav>\";\n\
\n\
\t// Check if a disconnected checkbox will retain its checked\n\
\t// value of true after appended to the DOM (IE6/7)\n\
\tinput.type = \"checkbox\";\n\
\tinput.checked = true;\n\
\tfragment.appendChild( input );\n\
\tsupport.appendChecked = input.checked;\n\
\n\
\t// Make sure textarea (and checkbox) defaultValue is properly cloned\n\
\t// Support: IE6-IE11+\n\
\tdiv.innerHTML = \"<textarea>x</textarea>\";\n\
\tsupport.noCloneChecked = !!div.cloneNode( true ).lastChild.defaultValue;\n\
\n\
\t// #11217 - WebKit loses check when the name is after the checked attribute\n\
\tfragment.appendChild( div );\n\
\tdiv.innerHTML = \"<input type='radio' checked='checked' name='t'/>\";\n\
\n\
\t// Support: Safari 5.1, iOS 5.1, Android 4.x, Android 2.3\n\
\t// old WebKit doesn't clone checked state correctly in fragments\n\
\tsupport.checkClone = div.cloneNode( true ).cloneNode( true ).lastChild.checked;\n\
\n\
\t// Support: IE<9\n\
\t// Opera does not clone events (and typeof div.attachEvent === undefined).\n\
\t// IE9-10 clones events bound via attachEvent, but they don't trigger with .click()\n\
\tsupport.noCloneEvent = true;\n\
\tif ( div.attachEvent ) {\n\
\t\tdiv.attachEvent( \"onclick\", function() {\n\
\t\t\tsupport.noCloneEvent = false;\n\
\t\t});\n\
\n\
\t\tdiv.cloneNode( true ).click();\n\
\t}\n\
\n\
\t// Execute the test only if not already executed in another module.\n\
\tif (support.deleteExpando == null) {\n\
\t\t// Support: IE<9\n\
\t\tsupport.deleteExpando = true;\n\
\t\ttry {\n\
\t\t\tdelete div.test;\n\
\t\t} catch( e ) {\n\
\t\t\tsupport.deleteExpando = false;\n\
\t\t}\n\
\t}\n\
})();\n\
\n\
\n\
(function() {\n\
\tvar i, eventName,\n\
\t\tdiv = document.createElement( \"div\" );\n\
\n\
\t// Support: IE<9 (lack submit/change bubble), Firefox 23+ (lack focusin event)\n\
\tfor ( i in { submit: true, change: true, focusin: true }) {\n\
\t\teventName = \"on\" + i;\n\
\n\
\t\tif ( !(support[ i + \"Bubbles\" ] = eventName in window) ) {\n\
\t\t\t// Beware of CSP restrictions (https://developer.mozilla.org/en/Security/CSP)\n\
\t\t\tdiv.setAttribute( eventName, \"t\" );\n\
\t\t\tsupport[ i + \"Bubbles\" ] = div.attributes[ eventName ].expando === false;\n\
\t\t}\n\
\t}\n\
\n\
\t// Null elements to avoid leaks in IE.\n\
\tdiv = null;\n\
})();\n\
\n\
\n\
var rformElems = /^(?:input|select|textarea)$/i,\n\
\trkeyEvent = /^key/,\n\
\trmouseEvent = /^(?:mouse|pointer|contextmenu)|click/,\n\
\trfocusMorph = /^(?:focusinfocus|focusoutblur)$/,\n\
\trtypenamespace = /^([^.]*)(?:\\.(.+)|)$/;\n\
\n\
function returnTrue() {\n\
\treturn true;\n\
}\n\
\n\
function returnFalse() {\n\
\treturn false;\n\
}\n\
\n\
function safeActiveElement() {\n\
\ttry {\n\
\t\treturn document.activeElement;\n\
\t} catch ( err ) { }\n\
}\n\
\n\
/*\n\
 * Helper functions for managing events -- not part of the public interface.\n\
 * Props to Dean Edwards' addEvent library for many of the ideas.\n\
 */\n\
jQuery.event = {\n\
\n\
\tglobal: {},\n\
\n\
\tadd: function( elem, types, handler, data, selector ) {\n\
\t\tvar tmp, events, t, handleObjIn,\n\
\t\t\tspecial, eventHandle, handleObj,\n\
\t\t\thandlers, type, namespaces, origType,\n\
\t\t\telemData = jQuery._data( elem );\n\
\n\
\t\t// Don't attach events to noData or text/comment nodes (but allow plain objects)\n\
\t\tif ( !elemData ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\t// Caller can pass in an object of custom data in lieu of the handler\n\
\t\tif ( handler.handler ) {\n\
\t\t\thandleObjIn = handler;\n\
\t\t\thandler = handleObjIn.handler;\n\
\t\t\tselector = handleObjIn.selector;\n\
\t\t}\n\
\n\
\t\t// Make sure that the handler has a unique ID, used to find/remove it later\n\
\t\tif ( !handler.guid ) {\n\
\t\t\thandler.guid = jQuery.guid++;\n\
\t\t}\n\
\n\
\t\t// Init the element's event structure and main handler, if this is the first\n\
\t\tif ( !(events = elemData.events) ) {\n\
\t\t\tevents = elemData.events = {};\n\
\t\t}\n\
\t\tif ( !(eventHandle = elemData.handle) ) {\n\
\t\t\teventHandle = elemData.handle = function( e ) {\n\
\t\t\t\t// Discard the second event of a jQuery.event.trigger() and\n\
\t\t\t\t// when an event is called after a page has unloaded\n\
\t\t\t\treturn typeof jQuery !== strundefined && (!e || jQuery.event.triggered !== e.type) ?\n\
\t\t\t\t\tjQuery.event.dispatch.apply( eventHandle.elem, arguments ) :\n\
\t\t\t\t\tundefined;\n\
\t\t\t};\n\
\t\t\t// Add elem as a property of the handle fn to prevent a memory leak with IE non-native events\n\
\t\t\teventHandle.elem = elem;\n\
\t\t}\n\
\n\
\t\t// Handle multiple events separated by a space\n\
\t\ttypes = ( types || \"\" ).match( rnotwhite ) || [ \"\" ];\n\
\t\tt = types.length;\n\
\t\twhile ( t-- ) {\n\
\t\t\ttmp = rtypenamespace.exec( types[t] ) || [];\n\
\t\t\ttype = origType = tmp[1];\n\
\t\t\tnamespaces = ( tmp[2] || \"\" ).split( \".\" ).sort();\n\
\n\
\t\t\t// There *must* be a type, no attaching namespace-only handlers\n\
\t\t\tif ( !type ) {\n\
\t\t\t\tcontinue;\n\
\t\t\t}\n\
\n\
\t\t\t// If event changes its type, use the special event handlers for the changed type\n\
\t\t\tspecial = jQuery.event.special[ type ] || {};\n\
\n\
\t\t\t// If selector defined, determine special event api type, otherwise given type\n\
\t\t\ttype = ( selector ? special.delegateType : special.bindType ) || type;\n\
\n\
\t\t\t// Update special based on newly reset type\n\
\t\t\tspecial = jQuery.event.special[ type ] || {};\n\
\n\
\t\t\t// handleObj is passed to all event handlers\n\
\t\t\thandleObj = jQuery.extend({\n\
\t\t\t\ttype: type,\n\
\t\t\t\torigType: origType,\n\
\t\t\t\tdata: data,\n\
\t\t\t\thandler: handler,\n\
\t\t\t\tguid: handler.guid,\n\
\t\t\t\tselector: selector,\n\
\t\t\t\tneedsContext: selector && jQuery.expr.match.needsContext.test( selector ),\n\
\t\t\t\tnamespace: namespaces.join(\".\")\n\
\t\t\t}, handleObjIn );\n\
\n\
\t\t\t// Init the event handler queue if we're the first\n\
\t\t\tif ( !(handlers = events[ type ]) ) {\n\
\t\t\t\thandlers = events[ type ] = [];\n\
\t\t\t\thandlers.delegateCount = 0;\n\
\n\
\t\t\t\t// Only use addEventListener/attachEvent if the special events handler returns false\n\
\t\t\t\tif ( !special.setup || special.setup.call( elem, data, namespaces, eventHandle ) === false ) {\n\
\t\t\t\t\t// Bind the global event handler to the element\n\
\t\t\t\t\tif ( elem.addEventListener ) {\n\
\t\t\t\t\t\telem.addEventListener( type, eventHandle, false );\n\
\n\
\t\t\t\t\t} else if ( elem.attachEvent ) {\n\
\t\t\t\t\t\telem.attachEvent( \"on\" + type, eventHandle );\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t\tif ( special.add ) {\n\
\t\t\t\tspecial.add.call( elem, handleObj );\n\
\n\
\t\t\t\tif ( !handleObj.handler.guid ) {\n\
\t\t\t\t\thandleObj.handler.guid = handler.guid;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t\t// Add to the element's handler list, delegates in front\n\
\t\t\tif ( selector ) {\n\
\t\t\t\thandlers.splice( handlers.delegateCount++, 0, handleObj );\n\
\t\t\t} else {\n\
\t\t\t\thandlers.push( handleObj );\n\
\t\t\t}\n\
\n\
\t\t\t// Keep track of which events have ever been used, for event optimization\n\
\t\t\tjQuery.event.global[ type ] = true;\n\
\t\t}\n\
\n\
\t\t// Nullify elem to prevent memory leaks in IE\n\
\t\telem = null;\n\
\t},\n\
\n\
\t// Detach an event or set of events from an element\n\
\tremove: function( elem, types, handler, selector, mappedTypes ) {\n\
\t\tvar j, handleObj, tmp,\n\
\t\t\torigCount, t, events,\n\
\t\t\tspecial, handlers, type,\n\
\t\t\tnamespaces, origType,\n\
\t\t\telemData = jQuery.hasData( elem ) && jQuery._data( elem );\n\
\n\
\t\tif ( !elemData || !(events = elemData.events) ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\t// Once for each type.namespace in types; type may be omitted\n\
\t\ttypes = ( types || \"\" ).match( rnotwhite ) || [ \"\" ];\n\
\t\tt = types.length;\n\
\t\twhile ( t-- ) {\n\
\t\t\ttmp = rtypenamespace.exec( types[t] ) || [];\n\
\t\t\ttype = origType = tmp[1];\n\
\t\t\tnamespaces = ( tmp[2] || \"\" ).split( \".\" ).sort();\n\
\n\
\t\t\t// Unbind all events (on this namespace, if provided) for the element\n\
\t\t\tif ( !type ) {\n\
\t\t\t\tfor ( type in events ) {\n\
\t\t\t\t\tjQuery.event.remove( elem, type + types[ t ], handler, selector, true );\n\
\t\t\t\t}\n\
\t\t\t\tcontinue;\n\
\t\t\t}\n\
\n\
\t\t\tspecial = jQuery.event.special[ type ] || {};\n\
\t\t\ttype = ( selector ? special.delegateType : special.bindType ) || type;\n\
\t\t\thandlers = events[ type ] || [];\n\
\t\t\ttmp = tmp[2] && new RegExp( \"(^|\\\\.)\" + namespaces.join(\"\\\\.(?:.*\\\\.|)\") + \"(\\\\.|$)\" );\n\
\n\
\t\t\t// Remove matching events\n\
\t\t\torigCount = j = handlers.length;\n\
\t\t\twhile ( j-- ) {\n\
\t\t\t\thandleObj = handlers[ j ];\n\
\n\
\t\t\t\tif ( ( mappedTypes || origType === handleObj.origType ) &&\n\
\t\t\t\t\t( !handler || handler.guid === handleObj.guid ) &&\n\
\t\t\t\t\t( !tmp || tmp.test( handleObj.namespace ) ) &&\n\
\t\t\t\t\t( !selector || selector === handleObj.selector || selector === \"**\" && handleObj.selector ) ) {\n\
\t\t\t\t\thandlers.splice( j, 1 );\n\
\n\
\t\t\t\t\tif ( handleObj.selector ) {\n\
\t\t\t\t\t\thandlers.delegateCount--;\n\
\t\t\t\t\t}\n\
\t\t\t\t\tif ( special.remove ) {\n\
\t\t\t\t\t\tspecial.remove.call( elem, handleObj );\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t\t// Remove generic event handler if we removed something and no more handlers exist\n\
\t\t\t// (avoids potential for endless recursion during removal of special event handlers)\n\
\t\t\tif ( origCount && !handlers.length ) {\n\
\t\t\t\tif ( !special.teardown || special.teardown.call( elem, namespaces, elemData.handle ) === false ) {\n\
\t\t\t\t\tjQuery.removeEvent( elem, type, elemData.handle );\n\
\t\t\t\t}\n\
\n\
\t\t\t\tdelete events[ type ];\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Remove the expando if it's no longer used\n\
\t\tif ( jQuery.isEmptyObject( events ) ) {\n\
\t\t\tdelete elemData.handle;\n\
\n\
\t\t\t// removeData also checks for emptiness and clears the expando if empty\n\
\t\t\t// so use it instead of delete\n\
\t\t\tjQuery._removeData( elem, \"events\" );\n\
\t\t}\n\
\t},\n\
\n\
\ttrigger: function( event, data, elem, onlyHandlers ) {\n\
\t\tvar handle, ontype, cur,\n\
\t\t\tbubbleType, special, tmp, i,\n\
\t\t\teventPath = [ elem || document ],\n\
\t\t\ttype = hasOwn.call( event, \"type\" ) ? event.type : event,\n\
\t\t\tnamespaces = hasOwn.call( event, \"namespace\" ) ? event.namespace.split(\".\") : [];\n\
\n\
\t\tcur = tmp = elem = elem || document;\n\
\n\
\t\t// Don't do events on text and comment nodes\n\
\t\tif ( elem.nodeType === 3 || elem.nodeType === 8 ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\t// focus/blur morphs to focusin/out; ensure we're not firing them right now\n\
\t\tif ( rfocusMorph.test( type + jQuery.event.triggered ) ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\tif ( type.indexOf(\".\") >= 0 ) {\n\
\t\t\t// Namespaced trigger; create a regexp to match event type in handle()\n\
\t\t\tnamespaces = type.split(\".\");\n\
\t\t\ttype = namespaces.shift();\n\
\t\t\tnamespaces.sort();\n\
\t\t}\n\
\t\tontype = type.indexOf(\":\") < 0 && \"on\" + type;\n\
\n\
\t\t// Caller can pass in a jQuery.Event object, Object, or just an event type string\n\
\t\tevent = event[ jQuery.expando ] ?\n\
\t\t\tevent :\n\
\t\t\tnew jQuery.Event( type, typeof event === \"object\" && event );\n\
\n\
\t\t// Trigger bitmask: & 1 for native handlers; & 2 for jQuery (always true)\n\
\t\tevent.isTrigger = onlyHandlers ? 2 : 3;\n\
\t\tevent.namespace = namespaces.join(\".\");\n\
\t\tevent.namespace_re = event.namespace ?\n\
\t\t\tnew RegExp( \"(^|\\\\.)\" + namespaces.join(\"\\\\.(?:.*\\\\.|)\") + \"(\\\\.|$)\" ) :\n\
\t\t\tnull;\n\
\n\
\t\t// Clean up the event in case it is being reused\n\
\t\tevent.result = undefined;\n\
\t\tif ( !event.target ) {\n\
\t\t\tevent.target = elem;\n\
\t\t}\n\
\n\
\t\t// Clone any incoming data and prepend the event, creating the handler arg list\n\
\t\tdata = data == null ?\n\
\t\t\t[ event ] :\n\
\t\t\tjQuery.makeArray( data, [ event ] );\n\
\n\
\t\t// Allow special events to draw outside the lines\n\
\t\tspecial = jQuery.event.special[ type ] || {};\n\
\t\tif ( !onlyHandlers && special.trigger && special.trigger.apply( elem, data ) === false ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\t// Determine event propagation path in advance, per W3C events spec (#9951)\n\
\t\t// Bubble up to document, then to window; watch for a global ownerDocument var (#9724)\n\
\t\tif ( !onlyHandlers && !special.noBubble && !jQuery.isWindow( elem ) ) {\n\
\n\
\t\t\tbubbleType = special.delegateType || type;\n\
\t\t\tif ( !rfocusMorph.test( bubbleType + type ) ) {\n\
\t\t\t\tcur = cur.parentNode;\n\
\t\t\t}\n\
\t\t\tfor ( ; cur; cur = cur.parentNode ) {\n\
\t\t\t\teventPath.push( cur );\n\
\t\t\t\ttmp = cur;\n\
\t\t\t}\n\
\n\
\t\t\t// Only add window if we got to document (e.g., not plain obj or detached DOM)\n\
\t\t\tif ( tmp === (elem.ownerDocument || document) ) {\n\
\t\t\t\teventPath.push( tmp.defaultView || tmp.parentWindow || window );\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Fire handlers on the event path\n\
\t\ti = 0;\n\
\t\twhile ( (cur = eventPath[i++]) && !event.isPropagationStopped() ) {\n\
\n\
\t\t\tevent.type = i > 1 ?\n\
\t\t\t\tbubbleType :\n\
\t\t\t\tspecial.bindType || type;\n\
\n\
\t\t\t// jQuery handler\n\
\t\t\thandle = ( jQuery._data( cur, \"events\" ) || {} )[ event.type ] && jQuery._data( cur, \"handle\" );\n\
\t\t\tif ( handle ) {\n\
\t\t\t\thandle.apply( cur, data );\n\
\t\t\t}\n\
\n\
\t\t\t// Native handler\n\
\t\t\thandle = ontype && cur[ ontype ];\n\
\t\t\tif ( handle && handle.apply && jQuery.acceptData( cur ) ) {\n\
\t\t\t\tevent.result = handle.apply( cur, data );\n\
\t\t\t\tif ( event.result === false ) {\n\
\t\t\t\t\tevent.preventDefault();\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\t\tevent.type = type;\n\
\n\
\t\t// If nobody prevented the default action, do it now\n\
\t\tif ( !onlyHandlers && !event.isDefaultPrevented() ) {\n\
\n\
\t\t\tif ( (!special._default || special._default.apply( eventPath.pop(), data ) === false) &&\n\
\t\t\t\tjQuery.acceptData( elem ) ) {\n\
\n\
\t\t\t\t// Call a native DOM method on the target with the same name name as the event.\n\
\t\t\t\t// Can't use an .isFunction() check here because IE6/7 fails that test.\n\
\t\t\t\t// Don't do default actions on window, that's where global variables be (#6170)\n\
\t\t\t\tif ( ontype && elem[ type ] && !jQuery.isWindow( elem ) ) {\n\
\n\
\t\t\t\t\t// Don't re-trigger an onFOO event when we call its FOO() method\n\
\t\t\t\t\ttmp = elem[ ontype ];\n\
\n\
\t\t\t\t\tif ( tmp ) {\n\
\t\t\t\t\t\telem[ ontype ] = null;\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// Prevent re-triggering of the same event, since we already bubbled it above\n\
\t\t\t\t\tjQuery.event.triggered = type;\n\
\t\t\t\t\ttry {\n\
\t\t\t\t\t\telem[ type ]();\n\
\t\t\t\t\t} catch ( e ) {\n\
\t\t\t\t\t\t// IE<9 dies on focus/blur to hidden element (#1486,#12518)\n\
\t\t\t\t\t\t// only reproducible on winXP IE8 native, not IE9 in IE8 mode\n\
\t\t\t\t\t}\n\
\t\t\t\t\tjQuery.event.triggered = undefined;\n\
\n\
\t\t\t\t\tif ( tmp ) {\n\
\t\t\t\t\t\telem[ ontype ] = tmp;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn event.result;\n\
\t},\n\
\n\
\tdispatch: function( event ) {\n\
\n\
\t\t// Make a writable jQuery.Event from the native event object\n\
\t\tevent = jQuery.event.fix( event );\n\
\n\
\t\tvar i, ret, handleObj, matched, j,\n\
\t\t\thandlerQueue = [],\n\
\t\t\targs = slice.call( arguments ),\n\
\t\t\thandlers = ( jQuery._data( this, \"events\" ) || {} )[ event.type ] || [],\n\
\t\t\tspecial = jQuery.event.special[ event.type ] || {};\n\
\n\
\t\t// Use the fix-ed jQuery.Event rather than the (read-only) native event\n\
\t\targs[0] = event;\n\
\t\tevent.delegateTarget = this;\n\
\n\
\t\t// Call the preDispatch hook for the mapped type, and let it bail if desired\n\
\t\tif ( special.preDispatch && special.preDispatch.call( this, event ) === false ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\t// Determine handlers\n\
\t\thandlerQueue = jQuery.event.handlers.call( this, event, handlers );\n\
\n\
\t\t// Run delegates first; they may want to stop propagation beneath us\n\
\t\ti = 0;\n\
\t\twhile ( (matched = handlerQueue[ i++ ]) && !event.isPropagationStopped() ) {\n\
\t\t\tevent.currentTarget = matched.elem;\n\
\n\
\t\t\tj = 0;\n\
\t\t\twhile ( (handleObj = matched.handlers[ j++ ]) && !event.isImmediatePropagationStopped() ) {\n\
\n\
\t\t\t\t// Triggered event must either 1) have no namespace, or\n\
\t\t\t\t// 2) have namespace(s) a subset or equal to those in the bound event (both can have no namespace).\n\
\t\t\t\tif ( !event.namespace_re || event.namespace_re.test( handleObj.namespace ) ) {\n\
\n\
\t\t\t\t\tevent.handleObj = handleObj;\n\
\t\t\t\t\tevent.data = handleObj.data;\n\
\n\
\t\t\t\t\tret = ( (jQuery.event.special[ handleObj.origType ] || {}).handle || handleObj.handler )\n\
\t\t\t\t\t\t\t.apply( matched.elem, args );\n\
\n\
\t\t\t\t\tif ( ret !== undefined ) {\n\
\t\t\t\t\t\tif ( (event.result = ret) === false ) {\n\
\t\t\t\t\t\t\tevent.preventDefault();\n\
\t\t\t\t\t\t\tevent.stopPropagation();\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Call the postDispatch hook for the mapped type\n\
\t\tif ( special.postDispatch ) {\n\
\t\t\tspecial.postDispatch.call( this, event );\n\
\t\t}\n\
\n\
\t\treturn event.result;\n\
\t},\n\
\n\
\thandlers: function( event, handlers ) {\n\
\t\tvar sel, handleObj, matches, i,\n\
\t\t\thandlerQueue = [],\n\
\t\t\tdelegateCount = handlers.delegateCount,\n\
\t\t\tcur = event.target;\n\
\n\
\t\t// Find delegate handlers\n\
\t\t// Black-hole SVG <use> instance trees (#13180)\n\
\t\t// Avoid non-left-click bubbling in Firefox (#3861)\n\
\t\tif ( delegateCount && cur.nodeType && (!event.button || event.type !== \"click\") ) {\n\
\n\
\t\t\t/* jshint eqeqeq: false */\n\
\t\t\tfor ( ; cur != this; cur = cur.parentNode || this ) {\n\
\t\t\t\t/* jshint eqeqeq: true */\n\
\n\
\t\t\t\t// Don't check non-elements (#13208)\n\
\t\t\t\t// Don't process clicks on disabled elements (#6911, #8165, #11382, #11764)\n\
\t\t\t\tif ( cur.nodeType === 1 && (cur.disabled !== true || event.type !== \"click\") ) {\n\
\t\t\t\t\tmatches = [];\n\
\t\t\t\t\tfor ( i = 0; i < delegateCount; i++ ) {\n\
\t\t\t\t\t\thandleObj = handlers[ i ];\n\
\n\
\t\t\t\t\t\t// Don't conflict with Object.prototype properties (#13203)\n\
\t\t\t\t\t\tsel = handleObj.selector + \" \";\n\
\n\
\t\t\t\t\t\tif ( matches[ sel ] === undefined ) {\n\
\t\t\t\t\t\t\tmatches[ sel ] = handleObj.needsContext ?\n\
\t\t\t\t\t\t\t\tjQuery( sel, this ).index( cur ) >= 0 :\n\
\t\t\t\t\t\t\t\tjQuery.find( sel, this, null, [ cur ] ).length;\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t\tif ( matches[ sel ] ) {\n\
\t\t\t\t\t\t\tmatches.push( handleObj );\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t\tif ( matches.length ) {\n\
\t\t\t\t\t\thandlerQueue.push({ elem: cur, handlers: matches });\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Add the remaining (directly-bound) handlers\n\
\t\tif ( delegateCount < handlers.length ) {\n\
\t\t\thandlerQueue.push({ elem: this, handlers: handlers.slice( delegateCount ) });\n\
\t\t}\n\
\n\
\t\treturn handlerQueue;\n\
\t},\n\
\n\
\tfix: function( event ) {\n\
\t\tif ( event[ jQuery.expando ] ) {\n\
\t\t\treturn event;\n\
\t\t}\n\
\n\
\t\t// Create a writable copy of the event object and normalize some properties\n\
\t\tvar i, prop, copy,\n\
\t\t\ttype = event.type,\n\
\t\t\toriginalEvent = event,\n\
\t\t\tfixHook = this.fixHooks[ type ];\n\
\n\
\t\tif ( !fixHook ) {\n\
\t\t\tthis.fixHooks[ type ] = fixHook =\n\
\t\t\t\trmouseEvent.test( type ) ? this.mouseHooks :\n\
\t\t\t\trkeyEvent.test( type ) ? this.keyHooks :\n\
\t\t\t\t{};\n\
\t\t}\n\
\t\tcopy = fixHook.props ? this.props.concat( fixHook.props ) : this.props;\n\
\n\
\t\tevent = new jQuery.Event( originalEvent );\n\
\n\
\t\ti = copy.length;\n\
\t\twhile ( i-- ) {\n\
\t\t\tprop = copy[ i ];\n\
\t\t\tevent[ prop ] = originalEvent[ prop ];\n\
\t\t}\n\
\n\
\t\t// Support: IE<9\n\
\t\t// Fix target property (#1925)\n\
\t\tif ( !event.target ) {\n\
\t\t\tevent.target = originalEvent.srcElement || document;\n\
\t\t}\n\
\n\
\t\t// Support: Chrome 23+, Safari?\n\
\t\t// Target should not be a text node (#504, #13143)\n\
\t\tif ( event.target.nodeType === 3 ) {\n\
\t\t\tevent.target = event.target.parentNode;\n\
\t\t}\n\
\n\
\t\t// Support: IE<9\n\
\t\t// For mouse/key events, metaKey==false if it's undefined (#3368, #11328)\n\
\t\tevent.metaKey = !!event.metaKey;\n\
\n\
\t\treturn fixHook.filter ? fixHook.filter( event, originalEvent ) : event;\n\
\t},\n\
\n\
\t// Includes some event props shared by KeyEvent and MouseEvent\n\
\tprops: \"altKey bubbles cancelable ctrlKey currentTarget eventPhase metaKey relatedTarget shiftKey target timeStamp view which\".split(\" \"),\n\
\n\
\tfixHooks: {},\n\
\n\
\tkeyHooks: {\n\
\t\tprops: \"char charCode key keyCode\".split(\" \"),\n\
\t\tfilter: function( event, original ) {\n\
\n\
\t\t\t// Add which for key events\n\
\t\t\tif ( event.which == null ) {\n\
\t\t\t\tevent.which = original.charCode != null ? original.charCode : original.keyCode;\n\
\t\t\t}\n\
\n\
\t\t\treturn event;\n\
\t\t}\n\
\t},\n\
\n\
\tmouseHooks: {\n\
\t\tprops: \"button buttons clientX clientY fromElement offsetX offsetY pageX pageY screenX screenY toElement\".split(\" \"),\n\
\t\tfilter: function( event, original ) {\n\
\t\t\tvar body, eventDoc, doc,\n\
\t\t\t\tbutton = original.button,\n\
\t\t\t\tfromElement = original.fromElement;\n\
\n\
\t\t\t// Calculate pageX/Y if missing and clientX/Y available\n\
\t\t\tif ( event.pageX == null && original.clientX != null ) {\n\
\t\t\t\teventDoc = event.target.ownerDocument || document;\n\
\t\t\t\tdoc = eventDoc.documentElement;\n\
\t\t\t\tbody = eventDoc.body;\n\
\n\
\t\t\t\tevent.pageX = original.clientX + ( doc && doc.scrollLeft || body && body.scrollLeft || 0 ) - ( doc && doc.clientLeft || body && body.clientLeft || 0 );\n\
\t\t\t\tevent.pageY = original.clientY + ( doc && doc.scrollTop  || body && body.scrollTop  || 0 ) - ( doc && doc.clientTop  || body && body.clientTop  || 0 );\n\
\t\t\t}\n\
\n\
\t\t\t// Add relatedTarget, if necessary\n\
\t\t\tif ( !event.relatedTarget && fromElement ) {\n\
\t\t\t\tevent.relatedTarget = fromElement === event.target ? original.toElement : fromElement;\n\
\t\t\t}\n\
\n\
\t\t\t// Add which for click: 1 === left; 2 === middle; 3 === right\n\
\t\t\t// Note: button is not normalized, so don't use it\n\
\t\t\tif ( !event.which && button !== undefined ) {\n\
\t\t\t\tevent.which = ( button & 1 ? 1 : ( button & 2 ? 3 : ( button & 4 ? 2 : 0 ) ) );\n\
\t\t\t}\n\
\n\
\t\t\treturn event;\n\
\t\t}\n\
\t},\n\
\n\
\tspecial: {\n\
\t\tload: {\n\
\t\t\t// Prevent triggered image.load events from bubbling to window.load\n\
\t\t\tnoBubble: true\n\
\t\t},\n\
\t\tfocus: {\n\
\t\t\t// Fire native event if possible so blur/focus sequence is correct\n\
\t\t\ttrigger: function() {\n\
\t\t\t\tif ( this !== safeActiveElement() && this.focus ) {\n\
\t\t\t\t\ttry {\n\
\t\t\t\t\t\tthis.focus();\n\
\t\t\t\t\t\treturn false;\n\
\t\t\t\t\t} catch ( e ) {\n\
\t\t\t\t\t\t// Support: IE<9\n\
\t\t\t\t\t\t// If we error on focus to hidden element (#1486, #12518),\n\
\t\t\t\t\t\t// let .trigger() run the handlers\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t},\n\
\t\t\tdelegateType: \"focusin\"\n\
\t\t},\n\
\t\tblur: {\n\
\t\t\ttrigger: function() {\n\
\t\t\t\tif ( this === safeActiveElement() && this.blur ) {\n\
\t\t\t\t\tthis.blur();\n\
\t\t\t\t\treturn false;\n\
\t\t\t\t}\n\
\t\t\t},\n\
\t\t\tdelegateType: \"focusout\"\n\
\t\t},\n\
\t\tclick: {\n\
\t\t\t// For checkbox, fire native event so checked state will be right\n\
\t\t\ttrigger: function() {\n\
\t\t\t\tif ( jQuery.nodeName( this, \"input\" ) && this.type === \"checkbox\" && this.click ) {\n\
\t\t\t\t\tthis.click();\n\
\t\t\t\t\treturn false;\n\
\t\t\t\t}\n\
\t\t\t},\n\
\n\
\t\t\t// For cross-browser consistency, don't fire native .click() on links\n\
\t\t\t_default: function( event ) {\n\
\t\t\t\treturn jQuery.nodeName( event.target, \"a\" );\n\
\t\t\t}\n\
\t\t},\n\
\n\
\t\tbeforeunload: {\n\
\t\t\tpostDispatch: function( event ) {\n\
\n\
\t\t\t\t// Support: Firefox 20+\n\
\t\t\t\t// Firefox doesn't alert if the returnValue field is not set.\n\
\t\t\t\tif ( event.result !== undefined && event.originalEvent ) {\n\
\t\t\t\t\tevent.originalEvent.returnValue = event.result;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\t},\n\
\n\
\tsimulate: function( type, elem, event, bubble ) {\n\
\t\t// Piggyback on a donor event to simulate a different one.\n\
\t\t// Fake originalEvent to avoid donor's stopPropagation, but if the\n\
\t\t// simulated event prevents default then we do the same on the donor.\n\
\t\tvar e = jQuery.extend(\n\
\t\t\tnew jQuery.Event(),\n\
\t\t\tevent,\n\
\t\t\t{\n\
\t\t\t\ttype: type,\n\
\t\t\t\tisSimulated: true,\n\
\t\t\t\toriginalEvent: {}\n\
\t\t\t}\n\
\t\t);\n\
\t\tif ( bubble ) {\n\
\t\t\tjQuery.event.trigger( e, null, elem );\n\
\t\t} else {\n\
\t\t\tjQuery.event.dispatch.call( elem, e );\n\
\t\t}\n\
\t\tif ( e.isDefaultPrevented() ) {\n\
\t\t\tevent.preventDefault();\n\
\t\t}\n\
\t}\n\
};\n\
\n\
jQuery.removeEvent = document.removeEventListener ?\n\
\tfunction( elem, type, handle ) {\n\
\t\tif ( elem.removeEventListener ) {\n\
\t\t\telem.removeEventListener( type, handle, false );\n\
\t\t}\n\
\t} :\n\
\tfunction( elem, type, handle ) {\n\
\t\tvar name = \"on\" + type;\n\
\n\
\t\tif ( elem.detachEvent ) {\n\
\n\
\t\t\t// #8545, #7054, preventing memory leaks for custom events in IE6-8\n\
\t\t\t// detachEvent needed property on element, by name of that event, to properly expose it to GC\n\
\t\t\tif ( typeof elem[ name ] === strundefined ) {\n\
\t\t\t\telem[ name ] = null;\n\
\t\t\t}\n\
\n\
\t\t\telem.detachEvent( name, handle );\n\
\t\t}\n\
\t};\n\
\n\
jQuery.Event = function( src, props ) {\n\
\t// Allow instantiation without the 'new' keyword\n\
\tif ( !(this instanceof jQuery.Event) ) {\n\
\t\treturn new jQuery.Event( src, props );\n\
\t}\n\
\n\
\t// Event object\n\
\tif ( src && src.type ) {\n\
\t\tthis.originalEvent = src;\n\
\t\tthis.type = src.type;\n\
\n\
\t\t// Events bubbling up the document may have been marked as prevented\n\
\t\t// by a handler lower down the tree; reflect the correct value.\n\
\t\tthis.isDefaultPrevented = src.defaultPrevented ||\n\
\t\t\t\tsrc.defaultPrevented === undefined &&\n\
\t\t\t\t// Support: IE < 9, Android < 4.0\n\
\t\t\t\tsrc.returnValue === false ?\n\
\t\t\treturnTrue :\n\
\t\t\treturnFalse;\n\
\n\
\t// Event type\n\
\t} else {\n\
\t\tthis.type = src;\n\
\t}\n\
\n\
\t// Put explicitly provided properties onto the event object\n\
\tif ( props ) {\n\
\t\tjQuery.extend( this, props );\n\
\t}\n\
\n\
\t// Create a timestamp if incoming event doesn't have one\n\
\tthis.timeStamp = src && src.timeStamp || jQuery.now();\n\
\n\
\t// Mark it as fixed\n\
\tthis[ jQuery.expando ] = true;\n\
};\n\
\n\
// jQuery.Event is based on DOM3 Events as specified by the ECMAScript Language Binding\n\
// http://www.w3.org/TR/2003/WD-DOM-Level-3-Events-20030331/ecma-script-binding.html\n\
jQuery.Event.prototype = {\n\
\tisDefaultPrevented: returnFalse,\n\
\tisPropagationStopped: returnFalse,\n\
\tisImmediatePropagationStopped: returnFalse,\n\
\n\
\tpreventDefault: function() {\n\
\t\tvar e = this.originalEvent;\n\
\n\
\t\tthis.isDefaultPrevented = returnTrue;\n\
\t\tif ( !e ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\t// If preventDefault exists, run it on the original event\n\
\t\tif ( e.preventDefault ) {\n\
\t\t\te.preventDefault();\n\
\n\
\t\t// Support: IE\n\
\t\t// Otherwise set the returnValue property of the original event to false\n\
\t\t} else {\n\
\t\t\te.returnValue = false;\n\
\t\t}\n\
\t},\n\
\tstopPropagation: function() {\n\
\t\tvar e = this.originalEvent;\n\
\n\
\t\tthis.isPropagationStopped = returnTrue;\n\
\t\tif ( !e ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\t\t// If stopPropagation exists, run it on the original event\n\
\t\tif ( e.stopPropagation ) {\n\
\t\t\te.stopPropagation();\n\
\t\t}\n\
\n\
\t\t// Support: IE\n\
\t\t// Set the cancelBubble property of the original event to true\n\
\t\te.cancelBubble = true;\n\
\t},\n\
\tstopImmediatePropagation: function() {\n\
\t\tvar e = this.originalEvent;\n\
\n\
\t\tthis.isImmediatePropagationStopped = returnTrue;\n\
\n\
\t\tif ( e && e.stopImmediatePropagation ) {\n\
\t\t\te.stopImmediatePropagation();\n\
\t\t}\n\
\n\
\t\tthis.stopPropagation();\n\
\t}\n\
};\n\
\n\
// Create mouseenter/leave events using mouseover/out and event-time checks\n\
jQuery.each({\n\
\tmouseenter: \"mouseover\",\n\
\tmouseleave: \"mouseout\",\n\
\tpointerenter: \"pointerover\",\n\
\tpointerleave: \"pointerout\"\n\
}, function( orig, fix ) {\n\
\tjQuery.event.special[ orig ] = {\n\
\t\tdelegateType: fix,\n\
\t\tbindType: fix,\n\
\n\
\t\thandle: function( event ) {\n\
\t\t\tvar ret,\n\
\t\t\t\ttarget = this,\n\
\t\t\t\trelated = event.relatedTarget,\n\
\t\t\t\thandleObj = event.handleObj;\n\
\n\
\t\t\t// For mousenter/leave call the handler if related is outside the target.\n\
\t\t\t// NB: No relatedTarget if the mouse left/entered the browser window\n\
\t\t\tif ( !related || (related !== target && !jQuery.contains( target, related )) ) {\n\
\t\t\t\tevent.type = handleObj.origType;\n\
\t\t\t\tret = handleObj.handler.apply( this, arguments );\n\
\t\t\t\tevent.type = fix;\n\
\t\t\t}\n\
\t\t\treturn ret;\n\
\t\t}\n\
\t};\n\
});\n\
\n\
// IE submit delegation\n\
if ( !support.submitBubbles ) {\n\
\n\
\tjQuery.event.special.submit = {\n\
\t\tsetup: function() {\n\
\t\t\t// Only need this for delegated form submit events\n\
\t\t\tif ( jQuery.nodeName( this, \"form\" ) ) {\n\
\t\t\t\treturn false;\n\
\t\t\t}\n\
\n\
\t\t\t// Lazy-add a submit handler when a descendant form may potentially be submitted\n\
\t\t\tjQuery.event.add( this, \"click._submit keypress._submit\", function( e ) {\n\
\t\t\t\t// Node name check avoids a VML-related crash in IE (#9807)\n\
\t\t\t\tvar elem = e.target,\n\
\t\t\t\t\tform = jQuery.nodeName( elem, \"input\" ) || jQuery.nodeName( elem, \"button\" ) ? elem.form : undefined;\n\
\t\t\t\tif ( form && !jQuery._data( form, \"submitBubbles\" ) ) {\n\
\t\t\t\t\tjQuery.event.add( form, \"submit._submit\", function( event ) {\n\
\t\t\t\t\t\tevent._submit_bubble = true;\n\
\t\t\t\t\t});\n\
\t\t\t\t\tjQuery._data( form, \"submitBubbles\", true );\n\
\t\t\t\t}\n\
\t\t\t});\n\
\t\t\t// return undefined since we don't need an event listener\n\
\t\t},\n\
\n\
\t\tpostDispatch: function( event ) {\n\
\t\t\t// If form was submitted by the user, bubble the event up the tree\n\
\t\t\tif ( event._submit_bubble ) {\n\
\t\t\t\tdelete event._submit_bubble;\n\
\t\t\t\tif ( this.parentNode && !event.isTrigger ) {\n\
\t\t\t\t\tjQuery.event.simulate( \"submit\", this.parentNode, event, true );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t},\n\
\n\
\t\tteardown: function() {\n\
\t\t\t// Only need this for delegated form submit events\n\
\t\t\tif ( jQuery.nodeName( this, \"form\" ) ) {\n\
\t\t\t\treturn false;\n\
\t\t\t}\n\
\n\
\t\t\t// Remove delegated handlers; cleanData eventually reaps submit handlers attached above\n\
\t\t\tjQuery.event.remove( this, \"._submit\" );\n\
\t\t}\n\
\t};\n\
}\n\
\n\
// IE change delegation and checkbox/radio fix\n\
if ( !support.changeBubbles ) {\n\
\n\
\tjQuery.event.special.change = {\n\
\n\
\t\tsetup: function() {\n\
\n\
\t\t\tif ( rformElems.test( this.nodeName ) ) {\n\
\t\t\t\t// IE doesn't fire change on a check/radio until blur; trigger it on click\n\
\t\t\t\t// after a propertychange. Eat the blur-change in special.change.handle.\n\
\t\t\t\t// This still fires onchange a second time for check/radio after blur.\n\
\t\t\t\tif ( this.type === \"checkbox\" || this.type === \"radio\" ) {\n\
\t\t\t\t\tjQuery.event.add( this, \"propertychange._change\", function( event ) {\n\
\t\t\t\t\t\tif ( event.originalEvent.propertyName === \"checked\" ) {\n\
\t\t\t\t\t\t\tthis._just_changed = true;\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t});\n\
\t\t\t\t\tjQuery.event.add( this, \"click._change\", function( event ) {\n\
\t\t\t\t\t\tif ( this._just_changed && !event.isTrigger ) {\n\
\t\t\t\t\t\t\tthis._just_changed = false;\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t\t// Allow triggered, simulated change events (#11500)\n\
\t\t\t\t\t\tjQuery.event.simulate( \"change\", this, event, true );\n\
\t\t\t\t\t});\n\
\t\t\t\t}\n\
\t\t\t\treturn false;\n\
\t\t\t}\n\
\t\t\t// Delegated event; lazy-add a change handler on descendant inputs\n\
\t\t\tjQuery.event.add( this, \"beforeactivate._change\", function( e ) {\n\
\t\t\t\tvar elem = e.target;\n\
\n\
\t\t\t\tif ( rformElems.test( elem.nodeName ) && !jQuery._data( elem, \"changeBubbles\" ) ) {\n\
\t\t\t\t\tjQuery.event.add( elem, \"change._change\", function( event ) {\n\
\t\t\t\t\t\tif ( this.parentNode && !event.isSimulated && !event.isTrigger ) {\n\
\t\t\t\t\t\t\tjQuery.event.simulate( \"change\", this.parentNode, event, true );\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t});\n\
\t\t\t\t\tjQuery._data( elem, \"changeBubbles\", true );\n\
\t\t\t\t}\n\
\t\t\t});\n\
\t\t},\n\
\n\
\t\thandle: function( event ) {\n\
\t\t\tvar elem = event.target;\n\
\n\
\t\t\t// Swallow native change events from checkbox/radio, we already triggered them above\n\
\t\t\tif ( this !== elem || event.isSimulated || event.isTrigger || (elem.type !== \"radio\" && elem.type !== \"checkbox\") ) {\n\
\t\t\t\treturn event.handleObj.handler.apply( this, arguments );\n\
\t\t\t}\n\
\t\t},\n\
\n\
\t\tteardown: function() {\n\
\t\t\tjQuery.event.remove( this, \"._change\" );\n\
\n\
\t\t\treturn !rformElems.test( this.nodeName );\n\
\t\t}\n\
\t};\n\
}\n\
\n\
// Create \"bubbling\" focus and blur events\n\
if ( !support.focusinBubbles ) {\n\
\tjQuery.each({ focus: \"focusin\", blur: \"focusout\" }, function( orig, fix ) {\n\
\n\
\t\t// Attach a single capturing handler on the document while someone wants focusin/focusout\n\
\t\tvar handler = function( event ) {\n\
\t\t\t\tjQuery.event.simulate( fix, event.target, jQuery.event.fix( event ), true );\n\
\t\t\t};\n\
\n\
\t\tjQuery.event.special[ fix ] = {\n\
\t\t\tsetup: function() {\n\
\t\t\t\tvar doc = this.ownerDocument || this,\n\
\t\t\t\t\tattaches = jQuery._data( doc, fix );\n\
\n\
\t\t\t\tif ( !attaches ) {\n\
\t\t\t\t\tdoc.addEventListener( orig, handler, true );\n\
\t\t\t\t}\n\
\t\t\t\tjQuery._data( doc, fix, ( attaches || 0 ) + 1 );\n\
\t\t\t},\n\
\t\t\tteardown: function() {\n\
\t\t\t\tvar doc = this.ownerDocument || this,\n\
\t\t\t\t\tattaches = jQuery._data( doc, fix ) - 1;\n\
\n\
\t\t\t\tif ( !attaches ) {\n\
\t\t\t\t\tdoc.removeEventListener( orig, handler, true );\n\
\t\t\t\t\tjQuery._removeData( doc, fix );\n\
\t\t\t\t} else {\n\
\t\t\t\t\tjQuery._data( doc, fix, attaches );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t};\n\
\t});\n\
}\n\
\n\
jQuery.fn.extend({\n\
\n\
\ton: function( types, selector, data, fn, /*INTERNAL*/ one ) {\n\
\t\tvar type, origFn;\n\
\n\
\t\t// Types can be a map of types/handlers\n\
\t\tif ( typeof types === \"object\" ) {\n\
\t\t\t// ( types-Object, selector, data )\n\
\t\t\tif ( typeof selector !== \"string\" ) {\n\
\t\t\t\t// ( types-Object, data )\n\
\t\t\t\tdata = data || selector;\n\
\t\t\t\tselector = undefined;\n\
\t\t\t}\n\
\t\t\tfor ( type in types ) {\n\
\t\t\t\tthis.on( type, selector, data, types[ type ], one );\n\
\t\t\t}\n\
\t\t\treturn this;\n\
\t\t}\n\
\n\
\t\tif ( data == null && fn == null ) {\n\
\t\t\t// ( types, fn )\n\
\t\t\tfn = selector;\n\
\t\t\tdata = selector = undefined;\n\
\t\t} else if ( fn == null ) {\n\
\t\t\tif ( typeof selector === \"string\" ) {\n\
\t\t\t\t// ( types, selector, fn )\n\
\t\t\t\tfn = data;\n\
\t\t\t\tdata = undefined;\n\
\t\t\t} else {\n\
\t\t\t\t// ( types, data, fn )\n\
\t\t\t\tfn = data;\n\
\t\t\t\tdata = selector;\n\
\t\t\t\tselector = undefined;\n\
\t\t\t}\n\
\t\t}\n\
\t\tif ( fn === false ) {\n\
\t\t\tfn = returnFalse;\n\
\t\t} else if ( !fn ) {\n\
\t\t\treturn this;\n\
\t\t}\n\
\n\
\t\tif ( one === 1 ) {\n\
\t\t\torigFn = fn;\n\
\t\t\tfn = function( event ) {\n\
\t\t\t\t// Can use an empty set, since event contains the info\n\
\t\t\t\tjQuery().off( event );\n\
\t\t\t\treturn origFn.apply( this, arguments );\n\
\t\t\t};\n\
\t\t\t// Use same guid so caller can remove using origFn\n\
\t\t\tfn.guid = origFn.guid || ( origFn.guid = jQuery.guid++ );\n\
\t\t}\n\
\t\treturn this.each( function() {\n\
\t\t\tjQuery.event.add( this, types, fn, data, selector );\n\
\t\t});\n\
\t},\n\
\tone: function( types, selector, data, fn ) {\n\
\t\treturn this.on( types, selector, data, fn, 1 );\n\
\t},\n\
\toff: function( types, selector, fn ) {\n\
\t\tvar handleObj, type;\n\
\t\tif ( types && types.preventDefault && types.handleObj ) {\n\
\t\t\t// ( event )  dispatched jQuery.Event\n\
\t\t\thandleObj = types.handleObj;\n\
\t\t\tjQuery( types.delegateTarget ).off(\n\
\t\t\t\thandleObj.namespace ? handleObj.origType + \".\" + handleObj.namespace : handleObj.origType,\n\
\t\t\t\thandleObj.selector,\n\
\t\t\t\thandleObj.handler\n\
\t\t\t);\n\
\t\t\treturn this;\n\
\t\t}\n\
\t\tif ( typeof types === \"object\" ) {\n\
\t\t\t// ( types-object [, selector] )\n\
\t\t\tfor ( type in types ) {\n\
\t\t\t\tthis.off( type, selector, types[ type ] );\n\
\t\t\t}\n\
\t\t\treturn this;\n\
\t\t}\n\
\t\tif ( selector === false || typeof selector === \"function\" ) {\n\
\t\t\t// ( types [, fn] )\n\
\t\t\tfn = selector;\n\
\t\t\tselector = undefined;\n\
\t\t}\n\
\t\tif ( fn === false ) {\n\
\t\t\tfn = returnFalse;\n\
\t\t}\n\
\t\treturn this.each(function() {\n\
\t\t\tjQuery.event.remove( this, types, fn, selector );\n\
\t\t});\n\
\t},\n\
\n\
\ttrigger: function( type, data ) {\n\
\t\treturn this.each(function() {\n\
\t\t\tjQuery.event.trigger( type, data, this );\n\
\t\t});\n\
\t},\n\
\ttriggerHandler: function( type, data ) {\n\
\t\tvar elem = this[0];\n\
\t\tif ( elem ) {\n\
\t\t\treturn jQuery.event.trigger( type, data, elem, true );\n\
\t\t}\n\
\t}\n\
});\n\
\n\
\n\
function createSafeFragment( document ) {\n\
\tvar list = nodeNames.split( \"|\" ),\n\
\t\tsafeFrag = document.createDocumentFragment();\n\
\n\
\tif ( safeFrag.createElement ) {\n\
\t\twhile ( list.length ) {\n\
\t\t\tsafeFrag.createElement(\n\
\t\t\t\tlist.pop()\n\
\t\t\t);\n\
\t\t}\n\
\t}\n\
\treturn safeFrag;\n\
}\n\
\n\
var nodeNames = \"abbr|article|aside|audio|bdi|canvas|data|datalist|details|figcaption|figure|footer|\" +\n\
\t\t\"header|hgroup|mark|meter|nav|output|progress|section|summary|time|video\",\n\
\trinlinejQuery = / jQuery\\d+=\"(?:null|\\d+)\"/g,\n\
\trnoshimcache = new RegExp(\"<(?:\" + nodeNames + \")[\\\\s/>]\", \"i\"),\n\
\trleadingWhitespace = /^\\s+/,\n\
\trxhtmlTag = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\\w:]+)[^>]*)\\/>/gi,\n\
\trtagName = /<([\\w:]+)/,\n\
\trtbody = /<tbody/i,\n\
\trhtml = /<|&#?\\w+;/,\n\
\trnoInnerhtml = /<(?:script|style|link)/i,\n\
\t// checked=\"checked\" or checked\n\
\trchecked = /checked\\s*(?:[^=]|=\\s*.checked.)/i,\n\
\trscriptType = /^$|\\/(?:java|ecma)script/i,\n\
\trscriptTypeMasked = /^true\\/(.*)/,\n\
\trcleanScript = /^\\s*<!(?:\\[CDATA\\[|--)|(?:\\]\\]|--)>\\s*$/g,\n\
\n\
\t// We have to close these tags to support XHTML (#13200)\n\
\twrapMap = {\n\
\t\toption: [ 1, \"<select multiple='multiple'>\", \"</select>\" ],\n\
\t\tlegend: [ 1, \"<fieldset>\", \"</fieldset>\" ],\n\
\t\tarea: [ 1, \"<map>\", \"</map>\" ],\n\
\t\tparam: [ 1, \"<object>\", \"</object>\" ],\n\
\t\tthead: [ 1, \"<table>\", \"</table>\" ],\n\
\t\ttr: [ 2, \"<table><tbody>\", \"</tbody></table>\" ],\n\
\t\tcol: [ 2, \"<table><tbody></tbody><colgroup>\", \"</colgroup></table>\" ],\n\
\t\ttd: [ 3, \"<table><tbody><tr>\", \"</tr></tbody></table>\" ],\n\
\n\
\t\t// IE6-8 can't serialize link, script, style, or any html5 (NoScope) tags,\n\
\t\t// unless wrapped in a div with non-breaking characters in front of it.\n\
\t\t_default: support.htmlSerialize ? [ 0, \"\", \"\" ] : [ 1, \"X<div>\", \"</div>\"  ]\n\
\t},\n\
\tsafeFragment = createSafeFragment( document ),\n\
\tfragmentDiv = safeFragment.appendChild( document.createElement(\"div\") );\n\
\n\
wrapMap.optgroup = wrapMap.option;\n\
wrapMap.tbody = wrapMap.tfoot = wrapMap.colgroup = wrapMap.caption = wrapMap.thead;\n\
wrapMap.th = wrapMap.td;\n\
\n\
function getAll( context, tag ) {\n\
\tvar elems, elem,\n\
\t\ti = 0,\n\
\t\tfound = typeof context.getElementsByTagName !== strundefined ? context.getElementsByTagName( tag || \"*\" ) :\n\
\t\t\ttypeof context.querySelectorAll !== strundefined ? context.querySelectorAll( tag || \"*\" ) :\n\
\t\t\tundefined;\n\
\n\
\tif ( !found ) {\n\
\t\tfor ( found = [], elems = context.childNodes || context; (elem = elems[i]) != null; i++ ) {\n\
\t\t\tif ( !tag || jQuery.nodeName( elem, tag ) ) {\n\
\t\t\t\tfound.push( elem );\n\
\t\t\t} else {\n\
\t\t\t\tjQuery.merge( found, getAll( elem, tag ) );\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\treturn tag === undefined || tag && jQuery.nodeName( context, tag ) ?\n\
\t\tjQuery.merge( [ context ], found ) :\n\
\t\tfound;\n\
}\n\
\n\
// Used in buildFragment, fixes the defaultChecked property\n\
function fixDefaultChecked( elem ) {\n\
\tif ( rcheckableType.test( elem.type ) ) {\n\
\t\telem.defaultChecked = elem.checked;\n\
\t}\n\
}\n\
\n\
// Support: IE<8\n\
// Manipulating tables requires a tbody\n\
function manipulationTarget( elem, content ) {\n\
\treturn jQuery.nodeName( elem, \"table\" ) &&\n\
\t\tjQuery.nodeName( content.nodeType !== 11 ? content : content.firstChild, \"tr\" ) ?\n\
\n\
\t\telem.getElementsByTagName(\"tbody\")[0] ||\n\
\t\t\telem.appendChild( elem.ownerDocument.createElement(\"tbody\") ) :\n\
\t\telem;\n\
}\n\
\n\
// Replace/restore the type attribute of script elements for safe DOM manipulation\n\
function disableScript( elem ) {\n\
\telem.type = (jQuery.find.attr( elem, \"type\" ) !== null) + \"/\" + elem.type;\n\
\treturn elem;\n\
}\n\
function restoreScript( elem ) {\n\
\tvar match = rscriptTypeMasked.exec( elem.type );\n\
\tif ( match ) {\n\
\t\telem.type = match[1];\n\
\t} else {\n\
\t\telem.removeAttribute(\"type\");\n\
\t}\n\
\treturn elem;\n\
}\n\
\n\
// Mark scripts as having already been evaluated\n\
function setGlobalEval( elems, refElements ) {\n\
\tvar elem,\n\
\t\ti = 0;\n\
\tfor ( ; (elem = elems[i]) != null; i++ ) {\n\
\t\tjQuery._data( elem, \"globalEval\", !refElements || jQuery._data( refElements[i], \"globalEval\" ) );\n\
\t}\n\
}\n\
\n\
function cloneCopyEvent( src, dest ) {\n\
\n\
\tif ( dest.nodeType !== 1 || !jQuery.hasData( src ) ) {\n\
\t\treturn;\n\
\t}\n\
\n\
\tvar type, i, l,\n\
\t\toldData = jQuery._data( src ),\n\
\t\tcurData = jQuery._data( dest, oldData ),\n\
\t\tevents = oldData.events;\n\
\n\
\tif ( events ) {\n\
\t\tdelete curData.handle;\n\
\t\tcurData.events = {};\n\
\n\
\t\tfor ( type in events ) {\n\
\t\t\tfor ( i = 0, l = events[ type ].length; i < l; i++ ) {\n\
\t\t\t\tjQuery.event.add( dest, type, events[ type ][ i ] );\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\t// make the cloned public data object a copy from the original\n\
\tif ( curData.data ) {\n\
\t\tcurData.data = jQuery.extend( {}, curData.data );\n\
\t}\n\
}\n\
\n\
function fixCloneNodeIssues( src, dest ) {\n\
\tvar nodeName, e, data;\n\
\n\
\t// We do not need to do anything for non-Elements\n\
\tif ( dest.nodeType !== 1 ) {\n\
\t\treturn;\n\
\t}\n\
\n\
\tnodeName = dest.nodeName.toLowerCase();\n\
\n\
\t// IE6-8 copies events bound via attachEvent when using cloneNode.\n\
\tif ( !support.noCloneEvent && dest[ jQuery.expando ] ) {\n\
\t\tdata = jQuery._data( dest );\n\
\n\
\t\tfor ( e in data.events ) {\n\
\t\t\tjQuery.removeEvent( dest, e, data.handle );\n\
\t\t}\n\
\n\
\t\t// Event data gets referenced instead of copied if the expando gets copied too\n\
\t\tdest.removeAttribute( jQuery.expando );\n\
\t}\n\
\n\
\t// IE blanks contents when cloning scripts, and tries to evaluate newly-set text\n\
\tif ( nodeName === \"script\" && dest.text !== src.text ) {\n\
\t\tdisableScript( dest ).text = src.text;\n\
\t\trestoreScript( dest );\n\
\n\
\t// IE6-10 improperly clones children of object elements using classid.\n\
\t// IE10 throws NoModificationAllowedError if parent is null, #12132.\n\
\t} else if ( nodeName === \"object\" ) {\n\
\t\tif ( dest.parentNode ) {\n\
\t\t\tdest.outerHTML = src.outerHTML;\n\
\t\t}\n\
\n\
\t\t// This path appears unavoidable for IE9. When cloning an object\n\
\t\t// element in IE9, the outerHTML strategy above is not sufficient.\n\
\t\t// If the src has innerHTML and the destination does not,\n\
\t\t// copy the src.innerHTML into the dest.innerHTML. #10324\n\
\t\tif ( support.html5Clone && ( src.innerHTML && !jQuery.trim(dest.innerHTML) ) ) {\n\
\t\t\tdest.innerHTML = src.innerHTML;\n\
\t\t}\n\
\n\
\t} else if ( nodeName === \"input\" && rcheckableType.test( src.type ) ) {\n\
\t\t// IE6-8 fails to persist the checked state of a cloned checkbox\n\
\t\t// or radio button. Worse, IE6-7 fail to give the cloned element\n\
\t\t// a checked appearance if the defaultChecked value isn't also set\n\
\n\
\t\tdest.defaultChecked = dest.checked = src.checked;\n\
\n\
\t\t// IE6-7 get confused and end up setting the value of a cloned\n\
\t\t// checkbox/radio button to an empty string instead of \"on\"\n\
\t\tif ( dest.value !== src.value ) {\n\
\t\t\tdest.value = src.value;\n\
\t\t}\n\
\n\
\t// IE6-8 fails to return the selected option to the default selected\n\
\t// state when cloning options\n\
\t} else if ( nodeName === \"option\" ) {\n\
\t\tdest.defaultSelected = dest.selected = src.defaultSelected;\n\
\n\
\t// IE6-8 fails to set the defaultValue to the correct value when\n\
\t// cloning other types of input fields\n\
\t} else if ( nodeName === \"input\" || nodeName === \"textarea\" ) {\n\
\t\tdest.defaultValue = src.defaultValue;\n\
\t}\n\
}\n\
\n\
jQuery.extend({\n\
\tclone: function( elem, dataAndEvents, deepDataAndEvents ) {\n\
\t\tvar destElements, node, clone, i, srcElements,\n\
\t\t\tinPage = jQuery.contains( elem.ownerDocument, elem );\n\
\n\
\t\tif ( support.html5Clone || jQuery.isXMLDoc(elem) || !rnoshimcache.test( \"<\" + elem.nodeName + \">\" ) ) {\n\
\t\t\tclone = elem.cloneNode( true );\n\
\n\
\t\t// IE<=8 does not properly clone detached, unknown element nodes\n\
\t\t} else {\n\
\t\t\tfragmentDiv.innerHTML = elem.outerHTML;\n\
\t\t\tfragmentDiv.removeChild( clone = fragmentDiv.firstChild );\n\
\t\t}\n\
\n\
\t\tif ( (!support.noCloneEvent || !support.noCloneChecked) &&\n\
\t\t\t\t(elem.nodeType === 1 || elem.nodeType === 11) && !jQuery.isXMLDoc(elem) ) {\n\
\n\
\t\t\t// We eschew Sizzle here for performance reasons: http://jsperf.com/getall-vs-sizzle/2\n\
\t\t\tdestElements = getAll( clone );\n\
\t\t\tsrcElements = getAll( elem );\n\
\n\
\t\t\t// Fix all IE cloning issues\n\
\t\t\tfor ( i = 0; (node = srcElements[i]) != null; ++i ) {\n\
\t\t\t\t// Ensure that the destination node is not null; Fixes #9587\n\
\t\t\t\tif ( destElements[i] ) {\n\
\t\t\t\t\tfixCloneNodeIssues( node, destElements[i] );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Copy the events from the original to the clone\n\
\t\tif ( dataAndEvents ) {\n\
\t\t\tif ( deepDataAndEvents ) {\n\
\t\t\t\tsrcElements = srcElements || getAll( elem );\n\
\t\t\t\tdestElements = destElements || getAll( clone );\n\
\n\
\t\t\t\tfor ( i = 0; (node = srcElements[i]) != null; i++ ) {\n\
\t\t\t\t\tcloneCopyEvent( node, destElements[i] );\n\
\t\t\t\t}\n\
\t\t\t} else {\n\
\t\t\t\tcloneCopyEvent( elem, clone );\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Preserve script evaluation history\n\
\t\tdestElements = getAll( clone, \"script\" );\n\
\t\tif ( destElements.length > 0 ) {\n\
\t\t\tsetGlobalEval( destElements, !inPage && getAll( elem, \"script\" ) );\n\
\t\t}\n\
\n\
\t\tdestElements = srcElements = node = null;\n\
\n\
\t\t// Return the cloned set\n\
\t\treturn clone;\n\
\t},\n\
\n\
\tbuildFragment: function( elems, context, scripts, selection ) {\n\
\t\tvar j, elem, contains,\n\
\t\t\ttmp, tag, tbody, wrap,\n\
\t\t\tl = elems.length,\n\
\n\
\t\t\t// Ensure a safe fragment\n\
\t\t\tsafe = createSafeFragment( context ),\n\
\n\
\t\t\tnodes = [],\n\
\t\t\ti = 0;\n\
\n\
\t\tfor ( ; i < l; i++ ) {\n\
\t\t\telem = elems[ i ];\n\
\n\
\t\t\tif ( elem || elem === 0 ) {\n\
\n\
\t\t\t\t// Add nodes directly\n\
\t\t\t\tif ( jQuery.type( elem ) === \"object\" ) {\n\
\t\t\t\t\tjQuery.merge( nodes, elem.nodeType ? [ elem ] : elem );\n\
\n\
\t\t\t\t// Convert non-html into a text node\n\
\t\t\t\t} else if ( !rhtml.test( elem ) ) {\n\
\t\t\t\t\tnodes.push( context.createTextNode( elem ) );\n\
\n\
\t\t\t\t// Convert html into DOM nodes\n\
\t\t\t\t} else {\n\
\t\t\t\t\ttmp = tmp || safe.appendChild( context.createElement(\"div\") );\n\
\n\
\t\t\t\t\t// Deserialize a standard representation\n\
\t\t\t\t\ttag = (rtagName.exec( elem ) || [ \"\", \"\" ])[ 1 ].toLowerCase();\n\
\t\t\t\t\twrap = wrapMap[ tag ] || wrapMap._default;\n\
\n\
\t\t\t\t\ttmp.innerHTML = wrap[1] + elem.replace( rxhtmlTag, \"<$1></$2>\" ) + wrap[2];\n\
\n\
\t\t\t\t\t// Descend through wrappers to the right content\n\
\t\t\t\t\tj = wrap[0];\n\
\t\t\t\t\twhile ( j-- ) {\n\
\t\t\t\t\t\ttmp = tmp.lastChild;\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// Manually add leading whitespace removed by IE\n\
\t\t\t\t\tif ( !support.leadingWhitespace && rleadingWhitespace.test( elem ) ) {\n\
\t\t\t\t\t\tnodes.push( context.createTextNode( rleadingWhitespace.exec( elem )[0] ) );\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// Remove IE's autoinserted <tbody> from table fragments\n\
\t\t\t\t\tif ( !support.tbody ) {\n\
\n\
\t\t\t\t\t\t// String was a <table>, *may* have spurious <tbody>\n\
\t\t\t\t\t\telem = tag === \"table\" && !rtbody.test( elem ) ?\n\
\t\t\t\t\t\t\ttmp.firstChild :\n\
\n\
\t\t\t\t\t\t\t// String was a bare <thead> or <tfoot>\n\
\t\t\t\t\t\t\twrap[1] === \"<table>\" && !rtbody.test( elem ) ?\n\
\t\t\t\t\t\t\t\ttmp :\n\
\t\t\t\t\t\t\t\t0;\n\
\n\
\t\t\t\t\t\tj = elem && elem.childNodes.length;\n\
\t\t\t\t\t\twhile ( j-- ) {\n\
\t\t\t\t\t\t\tif ( jQuery.nodeName( (tbody = elem.childNodes[j]), \"tbody\" ) && !tbody.childNodes.length ) {\n\
\t\t\t\t\t\t\t\telem.removeChild( tbody );\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\tjQuery.merge( nodes, tmp.childNodes );\n\
\n\
\t\t\t\t\t// Fix #12392 for WebKit and IE > 9\n\
\t\t\t\t\ttmp.textContent = \"\";\n\
\n\
\t\t\t\t\t// Fix #12392 for oldIE\n\
\t\t\t\t\twhile ( tmp.firstChild ) {\n\
\t\t\t\t\t\ttmp.removeChild( tmp.firstChild );\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// Remember the top-level container for proper cleanup\n\
\t\t\t\t\ttmp = safe.lastChild;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Fix #11356: Clear elements from fragment\n\
\t\tif ( tmp ) {\n\
\t\t\tsafe.removeChild( tmp );\n\
\t\t}\n\
\n\
\t\t// Reset defaultChecked for any radios and checkboxes\n\
\t\t// about to be appended to the DOM in IE 6/7 (#8060)\n\
\t\tif ( !support.appendChecked ) {\n\
\t\t\tjQuery.grep( getAll( nodes, \"input\" ), fixDefaultChecked );\n\
\t\t}\n\
\n\
\t\ti = 0;\n\
\t\twhile ( (elem = nodes[ i++ ]) ) {\n\
\n\
\t\t\t// #4087 - If origin and destination elements are the same, and this is\n\
\t\t\t// that element, do not do anything\n\
\t\t\tif ( selection && jQuery.inArray( elem, selection ) !== -1 ) {\n\
\t\t\t\tcontinue;\n\
\t\t\t}\n\
\n\
\t\t\tcontains = jQuery.contains( elem.ownerDocument, elem );\n\
\n\
\t\t\t// Append to fragment\n\
\t\t\ttmp = getAll( safe.appendChild( elem ), \"script\" );\n\
\n\
\t\t\t// Preserve script evaluation history\n\
\t\t\tif ( contains ) {\n\
\t\t\t\tsetGlobalEval( tmp );\n\
\t\t\t}\n\
\n\
\t\t\t// Capture executables\n\
\t\t\tif ( scripts ) {\n\
\t\t\t\tj = 0;\n\
\t\t\t\twhile ( (elem = tmp[ j++ ]) ) {\n\
\t\t\t\t\tif ( rscriptType.test( elem.type || \"\" ) ) {\n\
\t\t\t\t\t\tscripts.push( elem );\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\ttmp = null;\n\
\n\
\t\treturn safe;\n\
\t},\n\
\n\
\tcleanData: function( elems, /* internal */ acceptData ) {\n\
\t\tvar elem, type, id, data,\n\
\t\t\ti = 0,\n\
\t\t\tinternalKey = jQuery.expando,\n\
\t\t\tcache = jQuery.cache,\n\
\t\t\tdeleteExpando = support.deleteExpando,\n\
\t\t\tspecial = jQuery.event.special;\n\
\n\
\t\tfor ( ; (elem = elems[i]) != null; i++ ) {\n\
\t\t\tif ( acceptData || jQuery.acceptData( elem ) ) {\n\
\n\
\t\t\t\tid = elem[ internalKey ];\n\
\t\t\t\tdata = id && cache[ id ];\n\
\n\
\t\t\t\tif ( data ) {\n\
\t\t\t\t\tif ( data.events ) {\n\
\t\t\t\t\t\tfor ( type in data.events ) {\n\
\t\t\t\t\t\t\tif ( special[ type ] ) {\n\
\t\t\t\t\t\t\t\tjQuery.event.remove( elem, type );\n\
\n\
\t\t\t\t\t\t\t// This is a shortcut to avoid jQuery.event.remove's overhead\n\
\t\t\t\t\t\t\t} else {\n\
\t\t\t\t\t\t\t\tjQuery.removeEvent( elem, type, data.handle );\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// Remove cache only if it was not already removed by jQuery.event.remove\n\
\t\t\t\t\tif ( cache[ id ] ) {\n\
\n\
\t\t\t\t\t\tdelete cache[ id ];\n\
\n\
\t\t\t\t\t\t// IE does not allow us to delete expando properties from nodes,\n\
\t\t\t\t\t\t// nor does it have a removeAttribute function on Document nodes;\n\
\t\t\t\t\t\t// we must handle all of these cases\n\
\t\t\t\t\t\tif ( deleteExpando ) {\n\
\t\t\t\t\t\t\tdelete elem[ internalKey ];\n\
\n\
\t\t\t\t\t\t} else if ( typeof elem.removeAttribute !== strundefined ) {\n\
\t\t\t\t\t\t\telem.removeAttribute( internalKey );\n\
\n\
\t\t\t\t\t\t} else {\n\
\t\t\t\t\t\t\telem[ internalKey ] = null;\n\
\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t\tdeletedIds.push( id );\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
});\n\
\n\
jQuery.fn.extend({\n\
\ttext: function( value ) {\n\
\t\treturn access( this, function( value ) {\n\
\t\t\treturn value === undefined ?\n\
\t\t\t\tjQuery.text( this ) :\n\
\t\t\t\tthis.empty().append( ( this[0] && this[0].ownerDocument || document ).createTextNode( value ) );\n\
\t\t}, null, value, arguments.length );\n\
\t},\n\
\n\
\tappend: function() {\n\
\t\treturn this.domManip( arguments, function( elem ) {\n\
\t\t\tif ( this.nodeType === 1 || this.nodeType === 11 || this.nodeType === 9 ) {\n\
\t\t\t\tvar target = manipulationTarget( this, elem );\n\
\t\t\t\ttarget.appendChild( elem );\n\
\t\t\t}\n\
\t\t});\n\
\t},\n\
\n\
\tprepend: function() {\n\
\t\treturn this.domManip( arguments, function( elem ) {\n\
\t\t\tif ( this.nodeType === 1 || this.nodeType === 11 || this.nodeType === 9 ) {\n\
\t\t\t\tvar target = manipulationTarget( this, elem );\n\
\t\t\t\ttarget.insertBefore( elem, target.firstChild );\n\
\t\t\t}\n\
\t\t});\n\
\t},\n\
\n\
\tbefore: function() {\n\
\t\treturn this.domManip( arguments, function( elem ) {\n\
\t\t\tif ( this.parentNode ) {\n\
\t\t\t\tthis.parentNode.insertBefore( elem, this );\n\
\t\t\t}\n\
\t\t});\n\
\t},\n\
\n\
\tafter: function() {\n\
\t\treturn this.domManip( arguments, function( elem ) {\n\
\t\t\tif ( this.parentNode ) {\n\
\t\t\t\tthis.parentNode.insertBefore( elem, this.nextSibling );\n\
\t\t\t}\n\
\t\t});\n\
\t},\n\
\n\
\tremove: function( selector, keepData /* Internal Use Only */ ) {\n\
\t\tvar elem,\n\
\t\t\telems = selector ? jQuery.filter( selector, this ) : this,\n\
\t\t\ti = 0;\n\
\n\
\t\tfor ( ; (elem = elems[i]) != null; i++ ) {\n\
\n\
\t\t\tif ( !keepData && elem.nodeType === 1 ) {\n\
\t\t\t\tjQuery.cleanData( getAll( elem ) );\n\
\t\t\t}\n\
\n\
\t\t\tif ( elem.parentNode ) {\n\
\t\t\t\tif ( keepData && jQuery.contains( elem.ownerDocument, elem ) ) {\n\
\t\t\t\t\tsetGlobalEval( getAll( elem, \"script\" ) );\n\
\t\t\t\t}\n\
\t\t\t\telem.parentNode.removeChild( elem );\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn this;\n\
\t},\n\
\n\
\tempty: function() {\n\
\t\tvar elem,\n\
\t\t\ti = 0;\n\
\n\
\t\tfor ( ; (elem = this[i]) != null; i++ ) {\n\
\t\t\t// Remove element nodes and prevent memory leaks\n\
\t\t\tif ( elem.nodeType === 1 ) {\n\
\t\t\t\tjQuery.cleanData( getAll( elem, false ) );\n\
\t\t\t}\n\
\n\
\t\t\t// Remove any remaining nodes\n\
\t\t\twhile ( elem.firstChild ) {\n\
\t\t\t\telem.removeChild( elem.firstChild );\n\
\t\t\t}\n\
\n\
\t\t\t// If this is a select, ensure that it displays empty (#12336)\n\
\t\t\t// Support: IE<9\n\
\t\t\tif ( elem.options && jQuery.nodeName( elem, \"select\" ) ) {\n\
\t\t\t\telem.options.length = 0;\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn this;\n\
\t},\n\
\n\
\tclone: function( dataAndEvents, deepDataAndEvents ) {\n\
\t\tdataAndEvents = dataAndEvents == null ? false : dataAndEvents;\n\
\t\tdeepDataAndEvents = deepDataAndEvents == null ? dataAndEvents : deepDataAndEvents;\n\
\n\
\t\treturn this.map(function() {\n\
\t\t\treturn jQuery.clone( this, dataAndEvents, deepDataAndEvents );\n\
\t\t});\n\
\t},\n\
\n\
\thtml: function( value ) {\n\
\t\treturn access( this, function( value ) {\n\
\t\t\tvar elem = this[ 0 ] || {},\n\
\t\t\t\ti = 0,\n\
\t\t\t\tl = this.length;\n\
\n\
\t\t\tif ( value === undefined ) {\n\
\t\t\t\treturn elem.nodeType === 1 ?\n\
\t\t\t\t\telem.innerHTML.replace( rinlinejQuery, \"\" ) :\n\
\t\t\t\t\tundefined;\n\
\t\t\t}\n\
\n\
\t\t\t// See if we can take a shortcut and just use innerHTML\n\
\t\t\tif ( typeof value === \"string\" && !rnoInnerhtml.test( value ) &&\n\
\t\t\t\t( support.htmlSerialize || !rnoshimcache.test( value )  ) &&\n\
\t\t\t\t( support.leadingWhitespace || !rleadingWhitespace.test( value ) ) &&\n\
\t\t\t\t!wrapMap[ (rtagName.exec( value ) || [ \"\", \"\" ])[ 1 ].toLowerCase() ] ) {\n\
\n\
\t\t\t\tvalue = value.replace( rxhtmlTag, \"<$1></$2>\" );\n\
\n\
\t\t\t\ttry {\n\
\t\t\t\t\tfor (; i < l; i++ ) {\n\
\t\t\t\t\t\t// Remove element nodes and prevent memory leaks\n\
\t\t\t\t\t\telem = this[i] || {};\n\
\t\t\t\t\t\tif ( elem.nodeType === 1 ) {\n\
\t\t\t\t\t\t\tjQuery.cleanData( getAll( elem, false ) );\n\
\t\t\t\t\t\t\telem.innerHTML = value;\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\telem = 0;\n\
\n\
\t\t\t\t// If using innerHTML throws an exception, use the fallback method\n\
\t\t\t\t} catch(e) {}\n\
\t\t\t}\n\
\n\
\t\t\tif ( elem ) {\n\
\t\t\t\tthis.empty().append( value );\n\
\t\t\t}\n\
\t\t}, null, value, arguments.length );\n\
\t},\n\
\n\
\treplaceWith: function() {\n\
\t\tvar arg = arguments[ 0 ];\n\
\n\
\t\t// Make the changes, replacing each context element with the new content\n\
\t\tthis.domManip( arguments, function( elem ) {\n\
\t\t\targ = this.parentNode;\n\
\n\
\t\t\tjQuery.cleanData( getAll( this ) );\n\
\n\
\t\t\tif ( arg ) {\n\
\t\t\t\targ.replaceChild( elem, this );\n\
\t\t\t}\n\
\t\t});\n\
\n\
\t\t// Force removal if there was no new content (e.g., from empty arguments)\n\
\t\treturn arg && (arg.length || arg.nodeType) ? this : this.remove();\n\
\t},\n\
\n\
\tdetach: function( selector ) {\n\
\t\treturn this.remove( selector, true );\n\
\t},\n\
\n\
\tdomManip: function( args, callback ) {\n\
\n\
\t\t// Flatten any nested arrays\n\
\t\targs = concat.apply( [], args );\n\
\n\
\t\tvar first, node, hasScripts,\n\
\t\t\tscripts, doc, fragment,\n\
\t\t\ti = 0,\n\
\t\t\tl = this.length,\n\
\t\t\tset = this,\n\
\t\t\tiNoClone = l - 1,\n\
\t\t\tvalue = args[0],\n\
\t\t\tisFunction = jQuery.isFunction( value );\n\
\n\
\t\t// We can't cloneNode fragments that contain checked, in WebKit\n\
\t\tif ( isFunction ||\n\
\t\t\t\t( l > 1 && typeof value === \"string\" &&\n\
\t\t\t\t\t!support.checkClone && rchecked.test( value ) ) ) {\n\
\t\t\treturn this.each(function( index ) {\n\
\t\t\t\tvar self = set.eq( index );\n\
\t\t\t\tif ( isFunction ) {\n\
\t\t\t\t\targs[0] = value.call( this, index, self.html() );\n\
\t\t\t\t}\n\
\t\t\t\tself.domManip( args, callback );\n\
\t\t\t});\n\
\t\t}\n\
\n\
\t\tif ( l ) {\n\
\t\t\tfragment = jQuery.buildFragment( args, this[ 0 ].ownerDocument, false, this );\n\
\t\t\tfirst = fragment.firstChild;\n\
\n\
\t\t\tif ( fragment.childNodes.length === 1 ) {\n\
\t\t\t\tfragment = first;\n\
\t\t\t}\n\
\n\
\t\t\tif ( first ) {\n\
\t\t\t\tscripts = jQuery.map( getAll( fragment, \"script\" ), disableScript );\n\
\t\t\t\thasScripts = scripts.length;\n\
\n\
\t\t\t\t// Use the original fragment for the last item instead of the first because it can end up\n\
\t\t\t\t// being emptied incorrectly in certain situations (#8070).\n\
\t\t\t\tfor ( ; i < l; i++ ) {\n\
\t\t\t\t\tnode = fragment;\n\
\n\
\t\t\t\t\tif ( i !== iNoClone ) {\n\
\t\t\t\t\t\tnode = jQuery.clone( node, true, true );\n\
\n\
\t\t\t\t\t\t// Keep references to cloned scripts for later restoration\n\
\t\t\t\t\t\tif ( hasScripts ) {\n\
\t\t\t\t\t\t\tjQuery.merge( scripts, getAll( node, \"script\" ) );\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\tcallback.call( this[i], node, i );\n\
\t\t\t\t}\n\
\n\
\t\t\t\tif ( hasScripts ) {\n\
\t\t\t\t\tdoc = scripts[ scripts.length - 1 ].ownerDocument;\n\
\n\
\t\t\t\t\t// Reenable scripts\n\
\t\t\t\t\tjQuery.map( scripts, restoreScript );\n\
\n\
\t\t\t\t\t// Evaluate executable scripts on first document insertion\n\
\t\t\t\t\tfor ( i = 0; i < hasScripts; i++ ) {\n\
\t\t\t\t\t\tnode = scripts[ i ];\n\
\t\t\t\t\t\tif ( rscriptType.test( node.type || \"\" ) &&\n\
\t\t\t\t\t\t\t!jQuery._data( node, \"globalEval\" ) && jQuery.contains( doc, node ) ) {\n\
\n\
\t\t\t\t\t\t\tif ( node.src ) {\n\
\t\t\t\t\t\t\t\t// Optional AJAX dependency, but won't run scripts if not present\n\
\t\t\t\t\t\t\t\tif ( jQuery._evalUrl ) {\n\
\t\t\t\t\t\t\t\t\tjQuery._evalUrl( node.src );\n\
\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t} else {\n\
\t\t\t\t\t\t\t\tjQuery.globalEval( ( node.text || node.textContent || node.innerHTML || \"\" ).replace( rcleanScript, \"\" ) );\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\n\
\t\t\t\t// Fix #11809: Avoid leaking memory\n\
\t\t\t\tfragment = first = null;\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn this;\n\
\t}\n\
});\n\
\n\
jQuery.each({\n\
\tappendTo: \"append\",\n\
\tprependTo: \"prepend\",\n\
\tinsertBefore: \"before\",\n\
\tinsertAfter: \"after\",\n\
\treplaceAll: \"replaceWith\"\n\
}, function( name, original ) {\n\
\tjQuery.fn[ name ] = function( selector ) {\n\
\t\tvar elems,\n\
\t\t\ti = 0,\n\
\t\t\tret = [],\n\
\t\t\tinsert = jQuery( selector ),\n\
\t\t\tlast = insert.length - 1;\n\
\n\
\t\tfor ( ; i <= last; i++ ) {\n\
\t\t\telems = i === last ? this : this.clone(true);\n\
\t\t\tjQuery( insert[i] )[ original ]( elems );\n\
\n\
\t\t\t// Modern browsers can apply jQuery collections as arrays, but oldIE needs a .get()\n\
\t\t\tpush.apply( ret, elems.get() );\n\
\t\t}\n\
\n\
\t\treturn this.pushStack( ret );\n\
\t};\n\
});\n\
\n\
\n\
var iframe,\n\
\telemdisplay = {};\n\
\n\
/**\n\
 * Retrieve the actual display of a element\n\
 * @param {String} name nodeName of the element\n\
 * @param {Object} doc Document object\n\
 */\n\
// Called only from within defaultDisplay\n\
function actualDisplay( name, doc ) {\n\
\tvar style,\n\
\t\telem = jQuery( doc.createElement( name ) ).appendTo( doc.body ),\n\
\n\
\t\t// getDefaultComputedStyle might be reliably used only on attached element\n\
\t\tdisplay = window.getDefaultComputedStyle && ( style = window.getDefaultComputedStyle( elem[ 0 ] ) ) ?\n\
\n\
\t\t\t// Use of this method is a temporary fix (more like optmization) until something better comes along,\n\
\t\t\t// since it was removed from specification and supported only in FF\n\
\t\t\tstyle.display : jQuery.css( elem[ 0 ], \"display\" );\n\
\n\
\t// We don't have any data stored on the element,\n\
\t// so use \"detach\" method as fast way to get rid of the element\n\
\telem.detach();\n\
\n\
\treturn display;\n\
}\n\
\n\
/**\n\
 * Try to determine the default display value of an element\n\
 * @param {String} nodeName\n\
 */\n\
function defaultDisplay( nodeName ) {\n\
\tvar doc = document,\n\
\t\tdisplay = elemdisplay[ nodeName ];\n\
\n\
\tif ( !display ) {\n\
\t\tdisplay = actualDisplay( nodeName, doc );\n\
\n\
\t\t// If the simple way fails, read from inside an iframe\n\
\t\tif ( display === \"none\" || !display ) {\n\
\n\
\t\t\t// Use the already-created iframe if possible\n\
\t\t\tiframe = (iframe || jQuery( \"<iframe frameborder='0' width='0' height='0'/>\" )).appendTo( doc.documentElement );\n\
\n\
\t\t\t// Always write a new HTML skeleton so Webkit and Firefox don't choke on reuse\n\
\t\t\tdoc = ( iframe[ 0 ].contentWindow || iframe[ 0 ].contentDocument ).document;\n\
\n\
\t\t\t// Support: IE\n\
\t\t\tdoc.write();\n\
\t\t\tdoc.close();\n\
\n\
\t\t\tdisplay = actualDisplay( nodeName, doc );\n\
\t\t\tiframe.detach();\n\
\t\t}\n\
\n\
\t\t// Store the correct default display\n\
\t\telemdisplay[ nodeName ] = display;\n\
\t}\n\
\n\
\treturn display;\n\
}\n\
\n\
\n\
(function() {\n\
\tvar shrinkWrapBlocksVal;\n\
\n\
\tsupport.shrinkWrapBlocks = function() {\n\
\t\tif ( shrinkWrapBlocksVal != null ) {\n\
\t\t\treturn shrinkWrapBlocksVal;\n\
\t\t}\n\
\n\
\t\t// Will be changed later if needed.\n\
\t\tshrinkWrapBlocksVal = false;\n\
\n\
\t\t// Minified: var b,c,d\n\
\t\tvar div, body, container;\n\
\n\
\t\tbody = document.getElementsByTagName( \"body\" )[ 0 ];\n\
\t\tif ( !body || !body.style ) {\n\
\t\t\t// Test fired too early or in an unsupported environment, exit.\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\t// Setup\n\
\t\tdiv = document.createElement( \"div\" );\n\
\t\tcontainer = document.createElement( \"div\" );\n\
\t\tcontainer.style.cssText = \"position:absolute;border:0;width:0;height:0;top:0;left:-9999px\";\n\
\t\tbody.appendChild( container ).appendChild( div );\n\
\n\
\t\t// Support: IE6\n\
\t\t// Check if elements with layout shrink-wrap their children\n\
\t\tif ( typeof div.style.zoom !== strundefined ) {\n\
\t\t\t// Reset CSS: box-sizing; display; margin; border\n\
\t\t\tdiv.style.cssText =\n\
\t\t\t\t// Support: Firefox<29, Android 2.3\n\
\t\t\t\t// Vendor-prefix box-sizing\n\
\t\t\t\t\"-webkit-box-sizing:content-box;-moz-box-sizing:content-box;\" +\n\
\t\t\t\t\"box-sizing:content-box;display:block;margin:0;border:0;\" +\n\
\t\t\t\t\"padding:1px;width:1px;zoom:1\";\n\
\t\t\tdiv.appendChild( document.createElement( \"div\" ) ).style.width = \"5px\";\n\
\t\t\tshrinkWrapBlocksVal = div.offsetWidth !== 3;\n\
\t\t}\n\
\n\
\t\tbody.removeChild( container );\n\
\n\
\t\treturn shrinkWrapBlocksVal;\n\
\t};\n\
\n\
})();\n\
var rmargin = (/^margin/);\n\
\n\
var rnumnonpx = new RegExp( \"^(\" + pnum + \")(?!px)[a-z%]+$\", \"i\" );\n\
\n\
\n\
\n\
var getStyles, curCSS,\n\
\trposition = /^(top|right|bottom|left)$/;\n\
\n\
if ( window.getComputedStyle ) {\n\
\tgetStyles = function( elem ) {\n\
\t\treturn elem.ownerDocument.defaultView.getComputedStyle( elem, null );\n\
\t};\n\
\n\
\tcurCSS = function( elem, name, computed ) {\n\
\t\tvar width, minWidth, maxWidth, ret,\n\
\t\t\tstyle = elem.style;\n\
\n\
\t\tcomputed = computed || getStyles( elem );\n\
\n\
\t\t// getPropertyValue is only needed for .css('filter') in IE9, see #12537\n\
\t\tret = computed ? computed.getPropertyValue( name ) || computed[ name ] : undefined;\n\
\n\
\t\tif ( computed ) {\n\
\n\
\t\t\tif ( ret === \"\" && !jQuery.contains( elem.ownerDocument, elem ) ) {\n\
\t\t\t\tret = jQuery.style( elem, name );\n\
\t\t\t}\n\
\n\
\t\t\t// A tribute to the \"awesome hack by Dean Edwards\"\n\
\t\t\t// Chrome < 17 and Safari 5.0 uses \"computed value\" instead of \"used value\" for margin-right\n\
\t\t\t// Safari 5.1.7 (at least) returns percentage for a larger set of values, but width seems to be reliably pixels\n\
\t\t\t// this is against the CSSOM draft spec: http://dev.w3.org/csswg/cssom/#resolved-values\n\
\t\t\tif ( rnumnonpx.test( ret ) && rmargin.test( name ) ) {\n\
\n\
\t\t\t\t// Remember the original values\n\
\t\t\t\twidth = style.width;\n\
\t\t\t\tminWidth = style.minWidth;\n\
\t\t\t\tmaxWidth = style.maxWidth;\n\
\n\
\t\t\t\t// Put in the new values to get a computed value out\n\
\t\t\t\tstyle.minWidth = style.maxWidth = style.width = ret;\n\
\t\t\t\tret = computed.width;\n\
\n\
\t\t\t\t// Revert the changed values\n\
\t\t\t\tstyle.width = width;\n\
\t\t\t\tstyle.minWidth = minWidth;\n\
\t\t\t\tstyle.maxWidth = maxWidth;\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Support: IE\n\
\t\t// IE returns zIndex value as an integer.\n\
\t\treturn ret === undefined ?\n\
\t\t\tret :\n\
\t\t\tret + \"\";\n\
\t};\n\
} else if ( document.documentElement.currentStyle ) {\n\
\tgetStyles = function( elem ) {\n\
\t\treturn elem.currentStyle;\n\
\t};\n\
\n\
\tcurCSS = function( elem, name, computed ) {\n\
\t\tvar left, rs, rsLeft, ret,\n\
\t\t\tstyle = elem.style;\n\
\n\
\t\tcomputed = computed || getStyles( elem );\n\
\t\tret = computed ? computed[ name ] : undefined;\n\
\n\
\t\t// Avoid setting ret to empty string here\n\
\t\t// so we don't default to auto\n\
\t\tif ( ret == null && style && style[ name ] ) {\n\
\t\t\tret = style[ name ];\n\
\t\t}\n\
\n\
\t\t// From the awesome hack by Dean Edwards\n\
\t\t// http://erik.eae.net/archives/2007/07/27/18.54.15/#comment-102291\n\
\n\
\t\t// If we're not dealing with a regular pixel number\n\
\t\t// but a number that has a weird ending, we need to convert it to pixels\n\
\t\t// but not position css attributes, as those are proportional to the parent element instead\n\
\t\t// and we can't measure the parent instead because it might trigger a \"stacking dolls\" problem\n\
\t\tif ( rnumnonpx.test( ret ) && !rposition.test( name ) ) {\n\
\n\
\t\t\t// Remember the original values\n\
\t\t\tleft = style.left;\n\
\t\t\trs = elem.runtimeStyle;\n\
\t\t\trsLeft = rs && rs.left;\n\
\n\
\t\t\t// Put in the new values to get a computed value out\n\
\t\t\tif ( rsLeft ) {\n\
\t\t\t\trs.left = elem.currentStyle.left;\n\
\t\t\t}\n\
\t\t\tstyle.left = name === \"fontSize\" ? \"1em\" : ret;\n\
\t\t\tret = style.pixelLeft + \"px\";\n\
\n\
\t\t\t// Revert the changed values\n\
\t\t\tstyle.left = left;\n\
\t\t\tif ( rsLeft ) {\n\
\t\t\t\trs.left = rsLeft;\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Support: IE\n\
\t\t// IE returns zIndex value as an integer.\n\
\t\treturn ret === undefined ?\n\
\t\t\tret :\n\
\t\t\tret + \"\" || \"auto\";\n\
\t};\n\
}\n\
\n\
\n\
\n\
\n\
function addGetHookIf( conditionFn, hookFn ) {\n\
\t// Define the hook, we'll check on the first run if it's really needed.\n\
\treturn {\n\
\t\tget: function() {\n\
\t\t\tvar condition = conditionFn();\n\
\n\
\t\t\tif ( condition == null ) {\n\
\t\t\t\t// The test was not ready at this point; screw the hook this time\n\
\t\t\t\t// but check again when needed next time.\n\
\t\t\t\treturn;\n\
\t\t\t}\n\
\n\
\t\t\tif ( condition ) {\n\
\t\t\t\t// Hook not needed (or it's not possible to use it due to missing dependency),\n\
\t\t\t\t// remove it.\n\
\t\t\t\t// Since there are no other hooks for marginRight, remove the whole object.\n\
\t\t\t\tdelete this.get;\n\
\t\t\t\treturn;\n\
\t\t\t}\n\
\n\
\t\t\t// Hook needed; redefine it so that the support test is not executed again.\n\
\n\
\t\t\treturn (this.get = hookFn).apply( this, arguments );\n\
\t\t}\n\
\t};\n\
}\n\
\n\
\n\
(function() {\n\
\t// Minified: var b,c,d,e,f,g, h,i\n\
\tvar div, style, a, pixelPositionVal, boxSizingReliableVal,\n\
\t\treliableHiddenOffsetsVal, reliableMarginRightVal;\n\
\n\
\t// Setup\n\
\tdiv = document.createElement( \"div\" );\n\
\tdiv.innerHTML = \"  <link/><table></table><a href='/a'>a</a><input type='checkbox'/>\";\n\
\ta = div.getElementsByTagName( \"a\" )[ 0 ];\n\
\tstyle = a && a.style;\n\
\n\
\t// Finish early in limited (non-browser) environments\n\
\tif ( !style ) {\n\
\t\treturn;\n\
\t}\n\
\n\
\tstyle.cssText = \"float:left;opacity:.5\";\n\
\n\
\t// Support: IE<9\n\
\t// Make sure that element opacity exists (as opposed to filter)\n\
\tsupport.opacity = style.opacity === \"0.5\";\n\
\n\
\t// Verify style float existence\n\
\t// (IE uses styleFloat instead of cssFloat)\n\
\tsupport.cssFloat = !!style.cssFloat;\n\
\n\
\tdiv.style.backgroundClip = \"content-box\";\n\
\tdiv.cloneNode( true ).style.backgroundClip = \"\";\n\
\tsupport.clearCloneStyle = div.style.backgroundClip === \"content-box\";\n\
\n\
\t// Support: Firefox<29, Android 2.3\n\
\t// Vendor-prefix box-sizing\n\
\tsupport.boxSizing = style.boxSizing === \"\" || style.MozBoxSizing === \"\" ||\n\
\t\tstyle.WebkitBoxSizing === \"\";\n\
\n\
\tjQuery.extend(support, {\n\
\t\treliableHiddenOffsets: function() {\n\
\t\t\tif ( reliableHiddenOffsetsVal == null ) {\n\
\t\t\t\tcomputeStyleTests();\n\
\t\t\t}\n\
\t\t\treturn reliableHiddenOffsetsVal;\n\
\t\t},\n\
\n\
\t\tboxSizingReliable: function() {\n\
\t\t\tif ( boxSizingReliableVal == null ) {\n\
\t\t\t\tcomputeStyleTests();\n\
\t\t\t}\n\
\t\t\treturn boxSizingReliableVal;\n\
\t\t},\n\
\n\
\t\tpixelPosition: function() {\n\
\t\t\tif ( pixelPositionVal == null ) {\n\
\t\t\t\tcomputeStyleTests();\n\
\t\t\t}\n\
\t\t\treturn pixelPositionVal;\n\
\t\t},\n\
\n\
\t\t// Support: Android 2.3\n\
\t\treliableMarginRight: function() {\n\
\t\t\tif ( reliableMarginRightVal == null ) {\n\
\t\t\t\tcomputeStyleTests();\n\
\t\t\t}\n\
\t\t\treturn reliableMarginRightVal;\n\
\t\t}\n\
\t});\n\
\n\
\tfunction computeStyleTests() {\n\
\t\t// Minified: var b,c,d,j\n\
\t\tvar div, body, container, contents;\n\
\n\
\t\tbody = document.getElementsByTagName( \"body\" )[ 0 ];\n\
\t\tif ( !body || !body.style ) {\n\
\t\t\t// Test fired too early or in an unsupported environment, exit.\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\t// Setup\n\
\t\tdiv = document.createElement( \"div\" );\n\
\t\tcontainer = document.createElement( \"div\" );\n\
\t\tcontainer.style.cssText = \"position:absolute;border:0;width:0;height:0;top:0;left:-9999px\";\n\
\t\tbody.appendChild( container ).appendChild( div );\n\
\n\
\t\tdiv.style.cssText =\n\
\t\t\t// Support: Firefox<29, Android 2.3\n\
\t\t\t// Vendor-prefix box-sizing\n\
\t\t\t\"-webkit-box-sizing:border-box;-moz-box-sizing:border-box;\" +\n\
\t\t\t\"box-sizing:border-box;display:block;margin-top:1%;top:1%;\" +\n\
\t\t\t\"border:1px;padding:1px;width:4px;position:absolute\";\n\
\n\
\t\t// Support: IE<9\n\
\t\t// Assume reasonable values in the absence of getComputedStyle\n\
\t\tpixelPositionVal = boxSizingReliableVal = false;\n\
\t\treliableMarginRightVal = true;\n\
\n\
\t\t// Check for getComputedStyle so that this code is not run in IE<9.\n\
\t\tif ( window.getComputedStyle ) {\n\
\t\t\tpixelPositionVal = ( window.getComputedStyle( div, null ) || {} ).top !== \"1%\";\n\
\t\t\tboxSizingReliableVal =\n\
\t\t\t\t( window.getComputedStyle( div, null ) || { width: \"4px\" } ).width === \"4px\";\n\
\n\
\t\t\t// Support: Android 2.3\n\
\t\t\t// Div with explicit width and no margin-right incorrectly\n\
\t\t\t// gets computed margin-right based on width of container (#3333)\n\
\t\t\t// WebKit Bug 13343 - getComputedStyle returns wrong value for margin-right\n\
\t\t\tcontents = div.appendChild( document.createElement( \"div\" ) );\n\
\n\
\t\t\t// Reset CSS: box-sizing; display; margin; border; padding\n\
\t\t\tcontents.style.cssText = div.style.cssText =\n\
\t\t\t\t// Support: Firefox<29, Android 2.3\n\
\t\t\t\t// Vendor-prefix box-sizing\n\
\t\t\t\t\"-webkit-box-sizing:content-box;-moz-box-sizing:content-box;\" +\n\
\t\t\t\t\"box-sizing:content-box;display:block;margin:0;border:0;padding:0\";\n\
\t\t\tcontents.style.marginRight = contents.style.width = \"0\";\n\
\t\t\tdiv.style.width = \"1px\";\n\
\n\
\t\t\treliableMarginRightVal =\n\
\t\t\t\t!parseFloat( ( window.getComputedStyle( contents, null ) || {} ).marginRight );\n\
\t\t}\n\
\n\
\t\t// Support: IE8\n\
\t\t// Check if table cells still have offsetWidth/Height when they are set\n\
\t\t// to display:none and there are still other visible table cells in a\n\
\t\t// table row; if so, offsetWidth/Height are not reliable for use when\n\
\t\t// determining if an element has been hidden directly using\n\
\t\t// display:none (it is still safe to use offsets if a parent element is\n\
\t\t// hidden; don safety goggles and see bug #4512 for more information).\n\
\t\tdiv.innerHTML = \"<table><tr><td></td><td>t</td></tr></table>\";\n\
\t\tcontents = div.getElementsByTagName( \"td\" );\n\
\t\tcontents[ 0 ].style.cssText = \"margin:0;border:0;padding:0;display:none\";\n\
\t\treliableHiddenOffsetsVal = contents[ 0 ].offsetHeight === 0;\n\
\t\tif ( reliableHiddenOffsetsVal ) {\n\
\t\t\tcontents[ 0 ].style.display = \"\";\n\
\t\t\tcontents[ 1 ].style.display = \"none\";\n\
\t\t\treliableHiddenOffsetsVal = contents[ 0 ].offsetHeight === 0;\n\
\t\t}\n\
\n\
\t\tbody.removeChild( container );\n\
\t}\n\
\n\
})();\n\
\n\
\n\
// A method for quickly swapping in/out CSS properties to get correct calculations.\n\
jQuery.swap = function( elem, options, callback, args ) {\n\
\tvar ret, name,\n\
\t\told = {};\n\
\n\
\t// Remember the old values, and insert the new ones\n\
\tfor ( name in options ) {\n\
\t\told[ name ] = elem.style[ name ];\n\
\t\telem.style[ name ] = options[ name ];\n\
\t}\n\
\n\
\tret = callback.apply( elem, args || [] );\n\
\n\
\t// Revert the old values\n\
\tfor ( name in options ) {\n\
\t\telem.style[ name ] = old[ name ];\n\
\t}\n\
\n\
\treturn ret;\n\
};\n\
\n\
\n\
var\n\
\t\tralpha = /alpha\\([^)]*\\)/i,\n\
\tropacity = /opacity\\s*=\\s*([^)]*)/,\n\
\n\
\t// swappable if display is none or starts with table except \"table\", \"table-cell\", or \"table-caption\"\n\
\t// see here for display values: https://developer.mozilla.org/en-US/docs/CSS/display\n\
\trdisplayswap = /^(none|table(?!-c[ea]).+)/,\n\
\trnumsplit = new RegExp( \"^(\" + pnum + \")(.*)$\", \"i\" ),\n\
\trrelNum = new RegExp( \"^([+-])=(\" + pnum + \")\", \"i\" ),\n\
\n\
\tcssShow = { position: \"absolute\", visibility: \"hidden\", display: \"block\" },\n\
\tcssNormalTransform = {\n\
\t\tletterSpacing: \"0\",\n\
\t\tfontWeight: \"400\"\n\
\t},\n\
\n\
\tcssPrefixes = [ \"Webkit\", \"O\", \"Moz\", \"ms\" ];\n\
\n\
\n\
// return a css property mapped to a potentially vendor prefixed property\n\
function vendorPropName( style, name ) {\n\
\n\
\t// shortcut for names that are not vendor prefixed\n\
\tif ( name in style ) {\n\
\t\treturn name;\n\
\t}\n\
\n\
\t// check for vendor prefixed names\n\
\tvar capName = name.charAt(0).toUpperCase() + name.slice(1),\n\
\t\torigName = name,\n\
\t\ti = cssPrefixes.length;\n\
\n\
\twhile ( i-- ) {\n\
\t\tname = cssPrefixes[ i ] + capName;\n\
\t\tif ( name in style ) {\n\
\t\t\treturn name;\n\
\t\t}\n\
\t}\n\
\n\
\treturn origName;\n\
}\n\
\n\
function showHide( elements, show ) {\n\
\tvar display, elem, hidden,\n\
\t\tvalues = [],\n\
\t\tindex = 0,\n\
\t\tlength = elements.length;\n\
\n\
\tfor ( ; index < length; index++ ) {\n\
\t\telem = elements[ index ];\n\
\t\tif ( !elem.style ) {\n\
\t\t\tcontinue;\n\
\t\t}\n\
\n\
\t\tvalues[ index ] = jQuery._data( elem, \"olddisplay\" );\n\
\t\tdisplay = elem.style.display;\n\
\t\tif ( show ) {\n\
\t\t\t// Reset the inline display of this element to learn if it is\n\
\t\t\t// being hidden by cascaded rules or not\n\
\t\t\tif ( !values[ index ] && display === \"none\" ) {\n\
\t\t\t\telem.style.display = \"\";\n\
\t\t\t}\n\
\n\
\t\t\t// Set elements which have been overridden with display: none\n\
\t\t\t// in a stylesheet to whatever the default browser style is\n\
\t\t\t// for such an element\n\
\t\t\tif ( elem.style.display === \"\" && isHidden( elem ) ) {\n\
\t\t\t\tvalues[ index ] = jQuery._data( elem, \"olddisplay\", defaultDisplay(elem.nodeName) );\n\
\t\t\t}\n\
\t\t} else {\n\
\t\t\thidden = isHidden( elem );\n\
\n\
\t\t\tif ( display && display !== \"none\" || !hidden ) {\n\
\t\t\t\tjQuery._data( elem, \"olddisplay\", hidden ? display : jQuery.css( elem, \"display\" ) );\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\t// Set the display of most of the elements in a second loop\n\
\t// to avoid the constant reflow\n\
\tfor ( index = 0; index < length; index++ ) {\n\
\t\telem = elements[ index ];\n\
\t\tif ( !elem.style ) {\n\
\t\t\tcontinue;\n\
\t\t}\n\
\t\tif ( !show || elem.style.display === \"none\" || elem.style.display === \"\" ) {\n\
\t\t\telem.style.display = show ? values[ index ] || \"\" : \"none\";\n\
\t\t}\n\
\t}\n\
\n\
\treturn elements;\n\
}\n\
\n\
function setPositiveNumber( elem, value, subtract ) {\n\
\tvar matches = rnumsplit.exec( value );\n\
\treturn matches ?\n\
\t\t// Guard against undefined \"subtract\", e.g., when used as in cssHooks\n\
\t\tMath.max( 0, matches[ 1 ] - ( subtract || 0 ) ) + ( matches[ 2 ] || \"px\" ) :\n\
\t\tvalue;\n\
}\n\
\n\
function augmentWidthOrHeight( elem, name, extra, isBorderBox, styles ) {\n\
\tvar i = extra === ( isBorderBox ? \"border\" : \"content\" ) ?\n\
\t\t// If we already have the right measurement, avoid augmentation\n\
\t\t4 :\n\
\t\t// Otherwise initialize for horizontal or vertical properties\n\
\t\tname === \"width\" ? 1 : 0,\n\
\n\
\t\tval = 0;\n\
\n\
\tfor ( ; i < 4; i += 2 ) {\n\
\t\t// both box models exclude margin, so add it if we want it\n\
\t\tif ( extra === \"margin\" ) {\n\
\t\t\tval += jQuery.css( elem, extra + cssExpand[ i ], true, styles );\n\
\t\t}\n\
\n\
\t\tif ( isBorderBox ) {\n\
\t\t\t// border-box includes padding, so remove it if we want content\n\
\t\t\tif ( extra === \"content\" ) {\n\
\t\t\t\tval -= jQuery.css( elem, \"padding\" + cssExpand[ i ], true, styles );\n\
\t\t\t}\n\
\n\
\t\t\t// at this point, extra isn't border nor margin, so remove border\n\
\t\t\tif ( extra !== \"margin\" ) {\n\
\t\t\t\tval -= jQuery.css( elem, \"border\" + cssExpand[ i ] + \"Width\", true, styles );\n\
\t\t\t}\n\
\t\t} else {\n\
\t\t\t// at this point, extra isn't content, so add padding\n\
\t\t\tval += jQuery.css( elem, \"padding\" + cssExpand[ i ], true, styles );\n\
\n\
\t\t\t// at this point, extra isn't content nor padding, so add border\n\
\t\t\tif ( extra !== \"padding\" ) {\n\
\t\t\t\tval += jQuery.css( elem, \"border\" + cssExpand[ i ] + \"Width\", true, styles );\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\treturn val;\n\
}\n\
\n\
function getWidthOrHeight( elem, name, extra ) {\n\
\n\
\t// Start with offset property, which is equivalent to the border-box value\n\
\tvar valueIsBorderBox = true,\n\
\t\tval = name === \"width\" ? elem.offsetWidth : elem.offsetHeight,\n\
\t\tstyles = getStyles( elem ),\n\
\t\tisBorderBox = support.boxSizing && jQuery.css( elem, \"boxSizing\", false, styles ) === \"border-box\";\n\
\n\
\t// some non-html elements return undefined for offsetWidth, so check for null/undefined\n\
\t// svg - https://bugzilla.mozilla.org/show_bug.cgi?id=649285\n\
\t// MathML - https://bugzilla.mozilla.org/show_bug.cgi?id=491668\n\
\tif ( val <= 0 || val == null ) {\n\
\t\t// Fall back to computed then uncomputed css if necessary\n\
\t\tval = curCSS( elem, name, styles );\n\
\t\tif ( val < 0 || val == null ) {\n\
\t\t\tval = elem.style[ name ];\n\
\t\t}\n\
\n\
\t\t// Computed unit is not pixels. Stop here and return.\n\
\t\tif ( rnumnonpx.test(val) ) {\n\
\t\t\treturn val;\n\
\t\t}\n\
\n\
\t\t// we need the check for style in case a browser which returns unreliable values\n\
\t\t// for getComputedStyle silently falls back to the reliable elem.style\n\
\t\tvalueIsBorderBox = isBorderBox && ( support.boxSizingReliable() || val === elem.style[ name ] );\n\
\n\
\t\t// Normalize \"\", auto, and prepare for extra\n\
\t\tval = parseFloat( val ) || 0;\n\
\t}\n\
\n\
\t// use the active box-sizing model to add/subtract irrelevant styles\n\
\treturn ( val +\n\
\t\taugmentWidthOrHeight(\n\
\t\t\telem,\n\
\t\t\tname,\n\
\t\t\textra || ( isBorderBox ? \"border\" : \"content\" ),\n\
\t\t\tvalueIsBorderBox,\n\
\t\t\tstyles\n\
\t\t)\n\
\t) + \"px\";\n\
}\n\
\n\
jQuery.extend({\n\
\t// Add in style property hooks for overriding the default\n\
\t// behavior of getting and setting a style property\n\
\tcssHooks: {\n\
\t\topacity: {\n\
\t\t\tget: function( elem, computed ) {\n\
\t\t\t\tif ( computed ) {\n\
\t\t\t\t\t// We should always get a number back from opacity\n\
\t\t\t\t\tvar ret = curCSS( elem, \"opacity\" );\n\
\t\t\t\t\treturn ret === \"\" ? \"1\" : ret;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\t},\n\
\n\
\t// Don't automatically add \"px\" to these possibly-unitless properties\n\
\tcssNumber: {\n\
\t\t\"columnCount\": true,\n\
\t\t\"fillOpacity\": true,\n\
\t\t\"flexGrow\": true,\n\
\t\t\"flexShrink\": true,\n\
\t\t\"fontWeight\": true,\n\
\t\t\"lineHeight\": true,\n\
\t\t\"opacity\": true,\n\
\t\t\"order\": true,\n\
\t\t\"orphans\": true,\n\
\t\t\"widows\": true,\n\
\t\t\"zIndex\": true,\n\
\t\t\"zoom\": true\n\
\t},\n\
\n\
\t// Add in properties whose names you wish to fix before\n\
\t// setting or getting the value\n\
\tcssProps: {\n\
\t\t// normalize float css property\n\
\t\t\"float\": support.cssFloat ? \"cssFloat\" : \"styleFloat\"\n\
\t},\n\
\n\
\t// Get and set the style property on a DOM Node\n\
\tstyle: function( elem, name, value, extra ) {\n\
\t\t// Don't set styles on text and comment nodes\n\
\t\tif ( !elem || elem.nodeType === 3 || elem.nodeType === 8 || !elem.style ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\t// Make sure that we're working with the right name\n\
\t\tvar ret, type, hooks,\n\
\t\t\torigName = jQuery.camelCase( name ),\n\
\t\t\tstyle = elem.style;\n\
\n\
\t\tname = jQuery.cssProps[ origName ] || ( jQuery.cssProps[ origName ] = vendorPropName( style, origName ) );\n\
\n\
\t\t// gets hook for the prefixed version\n\
\t\t// followed by the unprefixed version\n\
\t\thooks = jQuery.cssHooks[ name ] || jQuery.cssHooks[ origName ];\n\
\n\
\t\t// Check if we're setting a value\n\
\t\tif ( value !== undefined ) {\n\
\t\t\ttype = typeof value;\n\
\n\
\t\t\t// convert relative number strings (+= or -=) to relative numbers. #7345\n\
\t\t\tif ( type === \"string\" && (ret = rrelNum.exec( value )) ) {\n\
\t\t\t\tvalue = ( ret[1] + 1 ) * ret[2] + parseFloat( jQuery.css( elem, name ) );\n\
\t\t\t\t// Fixes bug #9237\n\
\t\t\t\ttype = \"number\";\n\
\t\t\t}\n\
\n\
\t\t\t// Make sure that null and NaN values aren't set. See: #7116\n\
\t\t\tif ( value == null || value !== value ) {\n\
\t\t\t\treturn;\n\
\t\t\t}\n\
\n\
\t\t\t// If a number was passed in, add 'px' to the (except for certain CSS properties)\n\
\t\t\tif ( type === \"number\" && !jQuery.cssNumber[ origName ] ) {\n\
\t\t\t\tvalue += \"px\";\n\
\t\t\t}\n\
\n\
\t\t\t// Fixes #8908, it can be done more correctly by specifing setters in cssHooks,\n\
\t\t\t// but it would mean to define eight (for every problematic property) identical functions\n\
\t\t\tif ( !support.clearCloneStyle && value === \"\" && name.indexOf(\"background\") === 0 ) {\n\
\t\t\t\tstyle[ name ] = \"inherit\";\n\
\t\t\t}\n\
\n\
\t\t\t// If a hook was provided, use that value, otherwise just set the specified value\n\
\t\t\tif ( !hooks || !(\"set\" in hooks) || (value = hooks.set( elem, value, extra )) !== undefined ) {\n\
\n\
\t\t\t\t// Support: IE\n\
\t\t\t\t// Swallow errors from 'invalid' CSS values (#5509)\n\
\t\t\t\ttry {\n\
\t\t\t\t\tstyle[ name ] = value;\n\
\t\t\t\t} catch(e) {}\n\
\t\t\t}\n\
\n\
\t\t} else {\n\
\t\t\t// If a hook was provided get the non-computed value from there\n\
\t\t\tif ( hooks && \"get\" in hooks && (ret = hooks.get( elem, false, extra )) !== undefined ) {\n\
\t\t\t\treturn ret;\n\
\t\t\t}\n\
\n\
\t\t\t// Otherwise just get the value from the style object\n\
\t\t\treturn style[ name ];\n\
\t\t}\n\
\t},\n\
\n\
\tcss: function( elem, name, extra, styles ) {\n\
\t\tvar num, val, hooks,\n\
\t\t\torigName = jQuery.camelCase( name );\n\
\n\
\t\t// Make sure that we're working with the right name\n\
\t\tname = jQuery.cssProps[ origName ] || ( jQuery.cssProps[ origName ] = vendorPropName( elem.style, origName ) );\n\
\n\
\t\t// gets hook for the prefixed version\n\
\t\t// followed by the unprefixed version\n\
\t\thooks = jQuery.cssHooks[ name ] || jQuery.cssHooks[ origName ];\n\
\n\
\t\t// If a hook was provided get the computed value from there\n\
\t\tif ( hooks && \"get\" in hooks ) {\n\
\t\t\tval = hooks.get( elem, true, extra );\n\
\t\t}\n\
\n\
\t\t// Otherwise, if a way to get the computed value exists, use that\n\
\t\tif ( val === undefined ) {\n\
\t\t\tval = curCSS( elem, name, styles );\n\
\t\t}\n\
\n\
\t\t//convert \"normal\" to computed value\n\
\t\tif ( val === \"normal\" && name in cssNormalTransform ) {\n\
\t\t\tval = cssNormalTransform[ name ];\n\
\t\t}\n\
\n\
\t\t// Return, converting to number if forced or a qualifier was provided and val looks numeric\n\
\t\tif ( extra === \"\" || extra ) {\n\
\t\t\tnum = parseFloat( val );\n\
\t\t\treturn extra === true || jQuery.isNumeric( num ) ? num || 0 : val;\n\
\t\t}\n\
\t\treturn val;\n\
\t}\n\
});\n\
\n\
jQuery.each([ \"height\", \"width\" ], function( i, name ) {\n\
\tjQuery.cssHooks[ name ] = {\n\
\t\tget: function( elem, computed, extra ) {\n\
\t\t\tif ( computed ) {\n\
\t\t\t\t// certain elements can have dimension info if we invisibly show them\n\
\t\t\t\t// however, it must have a current display style that would benefit from this\n\
\t\t\t\treturn rdisplayswap.test( jQuery.css( elem, \"display\" ) ) && elem.offsetWidth === 0 ?\n\
\t\t\t\t\tjQuery.swap( elem, cssShow, function() {\n\
\t\t\t\t\t\treturn getWidthOrHeight( elem, name, extra );\n\
\t\t\t\t\t}) :\n\
\t\t\t\t\tgetWidthOrHeight( elem, name, extra );\n\
\t\t\t}\n\
\t\t},\n\
\n\
\t\tset: function( elem, value, extra ) {\n\
\t\t\tvar styles = extra && getStyles( elem );\n\
\t\t\treturn setPositiveNumber( elem, value, extra ?\n\
\t\t\t\taugmentWidthOrHeight(\n\
\t\t\t\t\telem,\n\
\t\t\t\t\tname,\n\
\t\t\t\t\textra,\n\
\t\t\t\t\tsupport.boxSizing && jQuery.css( elem, \"boxSizing\", false, styles ) === \"border-box\",\n\
\t\t\t\t\tstyles\n\
\t\t\t\t) : 0\n\
\t\t\t);\n\
\t\t}\n\
\t};\n\
});\n\
\n\
if ( !support.opacity ) {\n\
\tjQuery.cssHooks.opacity = {\n\
\t\tget: function( elem, computed ) {\n\
\t\t\t// IE uses filters for opacity\n\
\t\t\treturn ropacity.test( (computed && elem.currentStyle ? elem.currentStyle.filter : elem.style.filter) || \"\" ) ?\n\
\t\t\t\t( 0.01 * parseFloat( RegExp.$1 ) ) + \"\" :\n\
\t\t\t\tcomputed ? \"1\" : \"\";\n\
\t\t},\n\
\n\
\t\tset: function( elem, value ) {\n\
\t\t\tvar style = elem.style,\n\
\t\t\t\tcurrentStyle = elem.currentStyle,\n\
\t\t\t\topacity = jQuery.isNumeric( value ) ? \"alpha(opacity=\" + value * 100 + \")\" : \"\",\n\
\t\t\t\tfilter = currentStyle && currentStyle.filter || style.filter || \"\";\n\
\n\
\t\t\t// IE has trouble with opacity if it does not have layout\n\
\t\t\t// Force it by setting the zoom level\n\
\t\t\tstyle.zoom = 1;\n\
\n\
\t\t\t// if setting opacity to 1, and no other filters exist - attempt to remove filter attribute #6652\n\
\t\t\t// if value === \"\", then remove inline opacity #12685\n\
\t\t\tif ( ( value >= 1 || value === \"\" ) &&\n\
\t\t\t\t\tjQuery.trim( filter.replace( ralpha, \"\" ) ) === \"\" &&\n\
\t\t\t\t\tstyle.removeAttribute ) {\n\
\n\
\t\t\t\t// Setting style.filter to null, \"\" & \" \" still leave \"filter:\" in the cssText\n\
\t\t\t\t// if \"filter:\" is present at all, clearType is disabled, we want to avoid this\n\
\t\t\t\t// style.removeAttribute is IE Only, but so apparently is this code path...\n\
\t\t\t\tstyle.removeAttribute( \"filter\" );\n\
\n\
\t\t\t\t// if there is no filter style applied in a css rule or unset inline opacity, we are done\n\
\t\t\t\tif ( value === \"\" || currentStyle && !currentStyle.filter ) {\n\
\t\t\t\t\treturn;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t\t// otherwise, set new filter values\n\
\t\t\tstyle.filter = ralpha.test( filter ) ?\n\
\t\t\t\tfilter.replace( ralpha, opacity ) :\n\
\t\t\t\tfilter + \" \" + opacity;\n\
\t\t}\n\
\t};\n\
}\n\
\n\
jQuery.cssHooks.marginRight = addGetHookIf( support.reliableMarginRight,\n\
\tfunction( elem, computed ) {\n\
\t\tif ( computed ) {\n\
\t\t\t// WebKit Bug 13343 - getComputedStyle returns wrong value for margin-right\n\
\t\t\t// Work around by temporarily setting element display to inline-block\n\
\t\t\treturn jQuery.swap( elem, { \"display\": \"inline-block\" },\n\
\t\t\t\tcurCSS, [ elem, \"marginRight\" ] );\n\
\t\t}\n\
\t}\n\
);\n\
\n\
// These hooks are used by animate to expand properties\n\
jQuery.each({\n\
\tmargin: \"\",\n\
\tpadding: \"\",\n\
\tborder: \"Width\"\n\
}, function( prefix, suffix ) {\n\
\tjQuery.cssHooks[ prefix + suffix ] = {\n\
\t\texpand: function( value ) {\n\
\t\t\tvar i = 0,\n\
\t\t\t\texpanded = {},\n\
\n\
\t\t\t\t// assumes a single number if not a string\n\
\t\t\t\tparts = typeof value === \"string\" ? value.split(\" \") : [ value ];\n\
\n\
\t\t\tfor ( ; i < 4; i++ ) {\n\
\t\t\t\texpanded[ prefix + cssExpand[ i ] + suffix ] =\n\
\t\t\t\t\tparts[ i ] || parts[ i - 2 ] || parts[ 0 ];\n\
\t\t\t}\n\
\n\
\t\t\treturn expanded;\n\
\t\t}\n\
\t};\n\
\n\
\tif ( !rmargin.test( prefix ) ) {\n\
\t\tjQuery.cssHooks[ prefix + suffix ].set = setPositiveNumber;\n\
\t}\n\
});\n\
\n\
jQuery.fn.extend({\n\
\tcss: function( name, value ) {\n\
\t\treturn access( this, function( elem, name, value ) {\n\
\t\t\tvar styles, len,\n\
\t\t\t\tmap = {},\n\
\t\t\t\ti = 0;\n\
\n\
\t\t\tif ( jQuery.isArray( name ) ) {\n\
\t\t\t\tstyles = getStyles( elem );\n\
\t\t\t\tlen = name.length;\n\
\n\
\t\t\t\tfor ( ; i < len; i++ ) {\n\
\t\t\t\t\tmap[ name[ i ] ] = jQuery.css( elem, name[ i ], false, styles );\n\
\t\t\t\t}\n\
\n\
\t\t\t\treturn map;\n\
\t\t\t}\n\
\n\
\t\t\treturn value !== undefined ?\n\
\t\t\t\tjQuery.style( elem, name, value ) :\n\
\t\t\t\tjQuery.css( elem, name );\n\
\t\t}, name, value, arguments.length > 1 );\n\
\t},\n\
\tshow: function() {\n\
\t\treturn showHide( this, true );\n\
\t},\n\
\thide: function() {\n\
\t\treturn showHide( this );\n\
\t},\n\
\ttoggle: function( state ) {\n\
\t\tif ( typeof state === \"boolean\" ) {\n\
\t\t\treturn state ? this.show() : this.hide();\n\
\t\t}\n\
\n\
\t\treturn this.each(function() {\n\
\t\t\tif ( isHidden( this ) ) {\n\
\t\t\t\tjQuery( this ).show();\n\
\t\t\t} else {\n\
\t\t\t\tjQuery( this ).hide();\n\
\t\t\t}\n\
\t\t});\n\
\t}\n\
});\n\
\n\
\n\
function Tween( elem, options, prop, end, easing ) {\n\
\treturn new Tween.prototype.init( elem, options, prop, end, easing );\n\
}\n\
jQuery.Tween = Tween;\n\
\n\
Tween.prototype = {\n\
\tconstructor: Tween,\n\
\tinit: function( elem, options, prop, end, easing, unit ) {\n\
\t\tthis.elem = elem;\n\
\t\tthis.prop = prop;\n\
\t\tthis.easing = easing || \"swing\";\n\
\t\tthis.options = options;\n\
\t\tthis.start = this.now = this.cur();\n\
\t\tthis.end = end;\n\
\t\tthis.unit = unit || ( jQuery.cssNumber[ prop ] ? \"\" : \"px\" );\n\
\t},\n\
\tcur: function() {\n\
\t\tvar hooks = Tween.propHooks[ this.prop ];\n\
\n\
\t\treturn hooks && hooks.get ?\n\
\t\t\thooks.get( this ) :\n\
\t\t\tTween.propHooks._default.get( this );\n\
\t},\n\
\trun: function( percent ) {\n\
\t\tvar eased,\n\
\t\t\thooks = Tween.propHooks[ this.prop ];\n\
\n\
\t\tif ( this.options.duration ) {\n\
\t\t\tthis.pos = eased = jQuery.easing[ this.easing ](\n\
\t\t\t\tpercent, this.options.duration * percent, 0, 1, this.options.duration\n\
\t\t\t);\n\
\t\t} else {\n\
\t\t\tthis.pos = eased = percent;\n\
\t\t}\n\
\t\tthis.now = ( this.end - this.start ) * eased + this.start;\n\
\n\
\t\tif ( this.options.step ) {\n\
\t\t\tthis.options.step.call( this.elem, this.now, this );\n\
\t\t}\n\
\n\
\t\tif ( hooks && hooks.set ) {\n\
\t\t\thooks.set( this );\n\
\t\t} else {\n\
\t\t\tTween.propHooks._default.set( this );\n\
\t\t}\n\
\t\treturn this;\n\
\t}\n\
};\n\
\n\
Tween.prototype.init.prototype = Tween.prototype;\n\
\n\
Tween.propHooks = {\n\
\t_default: {\n\
\t\tget: function( tween ) {\n\
\t\t\tvar result;\n\
\n\
\t\t\tif ( tween.elem[ tween.prop ] != null &&\n\
\t\t\t\t(!tween.elem.style || tween.elem.style[ tween.prop ] == null) ) {\n\
\t\t\t\treturn tween.elem[ tween.prop ];\n\
\t\t\t}\n\
\n\
\t\t\t// passing an empty string as a 3rd parameter to .css will automatically\n\
\t\t\t// attempt a parseFloat and fallback to a string if the parse fails\n\
\t\t\t// so, simple values such as \"10px\" are parsed to Float.\n\
\t\t\t// complex values such as \"rotate(1rad)\" are returned as is.\n\
\t\t\tresult = jQuery.css( tween.elem, tween.prop, \"\" );\n\
\t\t\t// Empty strings, null, undefined and \"auto\" are converted to 0.\n\
\t\t\treturn !result || result === \"auto\" ? 0 : result;\n\
\t\t},\n\
\t\tset: function( tween ) {\n\
\t\t\t// use step hook for back compat - use cssHook if its there - use .style if its\n\
\t\t\t// available and use plain properties where available\n\
\t\t\tif ( jQuery.fx.step[ tween.prop ] ) {\n\
\t\t\t\tjQuery.fx.step[ tween.prop ]( tween );\n\
\t\t\t} else if ( tween.elem.style && ( tween.elem.style[ jQuery.cssProps[ tween.prop ] ] != null || jQuery.cssHooks[ tween.prop ] ) ) {\n\
\t\t\t\tjQuery.style( tween.elem, tween.prop, tween.now + tween.unit );\n\
\t\t\t} else {\n\
\t\t\t\ttween.elem[ tween.prop ] = tween.now;\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
};\n\
\n\
// Support: IE <=9\n\
// Panic based approach to setting things on disconnected nodes\n\
\n\
Tween.propHooks.scrollTop = Tween.propHooks.scrollLeft = {\n\
\tset: function( tween ) {\n\
\t\tif ( tween.elem.nodeType && tween.elem.parentNode ) {\n\
\t\t\ttween.elem[ tween.prop ] = tween.now;\n\
\t\t}\n\
\t}\n\
};\n\
\n\
jQuery.easing = {\n\
\tlinear: function( p ) {\n\
\t\treturn p;\n\
\t},\n\
\tswing: function( p ) {\n\
\t\treturn 0.5 - Math.cos( p * Math.PI ) / 2;\n\
\t}\n\
};\n\
\n\
jQuery.fx = Tween.prototype.init;\n\
\n\
// Back Compat <1.8 extension point\n\
jQuery.fx.step = {};\n\
\n\
\n\
\n\
\n\
var\n\
\tfxNow, timerId,\n\
\trfxtypes = /^(?:toggle|show|hide)$/,\n\
\trfxnum = new RegExp( \"^(?:([+-])=|)(\" + pnum + \")([a-z%]*)$\", \"i\" ),\n\
\trrun = /queueHooks$/,\n\
\tanimationPrefilters = [ defaultPrefilter ],\n\
\ttweeners = {\n\
\t\t\"*\": [ function( prop, value ) {\n\
\t\t\tvar tween = this.createTween( prop, value ),\n\
\t\t\t\ttarget = tween.cur(),\n\
\t\t\t\tparts = rfxnum.exec( value ),\n\
\t\t\t\tunit = parts && parts[ 3 ] || ( jQuery.cssNumber[ prop ] ? \"\" : \"px\" ),\n\
\n\
\t\t\t\t// Starting value computation is required for potential unit mismatches\n\
\t\t\t\tstart = ( jQuery.cssNumber[ prop ] || unit !== \"px\" && +target ) &&\n\
\t\t\t\t\trfxnum.exec( jQuery.css( tween.elem, prop ) ),\n\
\t\t\t\tscale = 1,\n\
\t\t\t\tmaxIterations = 20;\n\
\n\
\t\t\tif ( start && start[ 3 ] !== unit ) {\n\
\t\t\t\t// Trust units reported by jQuery.css\n\
\t\t\t\tunit = unit || start[ 3 ];\n\
\n\
\t\t\t\t// Make sure we update the tween properties later on\n\
\t\t\t\tparts = parts || [];\n\
\n\
\t\t\t\t// Iteratively approximate from a nonzero starting point\n\
\t\t\t\tstart = +target || 1;\n\
\n\
\t\t\t\tdo {\n\
\t\t\t\t\t// If previous iteration zeroed out, double until we get *something*\n\
\t\t\t\t\t// Use a string for doubling factor so we don't accidentally see scale as unchanged below\n\
\t\t\t\t\tscale = scale || \".5\";\n\
\n\
\t\t\t\t\t// Adjust and apply\n\
\t\t\t\t\tstart = start / scale;\n\
\t\t\t\t\tjQuery.style( tween.elem, prop, start + unit );\n\
\n\
\t\t\t\t// Update scale, tolerating zero or NaN from tween.cur()\n\
\t\t\t\t// And breaking the loop if scale is unchanged or perfect, or if we've just had enough\n\
\t\t\t\t} while ( scale !== (scale = tween.cur() / target) && scale !== 1 && --maxIterations );\n\
\t\t\t}\n\
\n\
\t\t\t// Update tween properties\n\
\t\t\tif ( parts ) {\n\
\t\t\t\tstart = tween.start = +start || +target || 0;\n\
\t\t\t\ttween.unit = unit;\n\
\t\t\t\t// If a +=/-= token was provided, we're doing a relative animation\n\
\t\t\t\ttween.end = parts[ 1 ] ?\n\
\t\t\t\t\tstart + ( parts[ 1 ] + 1 ) * parts[ 2 ] :\n\
\t\t\t\t\t+parts[ 2 ];\n\
\t\t\t}\n\
\n\
\t\t\treturn tween;\n\
\t\t} ]\n\
\t};\n\
\n\
// Animations created synchronously will run synchronously\n\
function createFxNow() {\n\
\tsetTimeout(function() {\n\
\t\tfxNow = undefined;\n\
\t});\n\
\treturn ( fxNow = jQuery.now() );\n\
}\n\
\n\
// Generate parameters to create a standard animation\n\
function genFx( type, includeWidth ) {\n\
\tvar which,\n\
\t\tattrs = { height: type },\n\
\t\ti = 0;\n\
\n\
\t// if we include width, step value is 1 to do all cssExpand values,\n\
\t// if we don't include width, step value is 2 to skip over Left and Right\n\
\tincludeWidth = includeWidth ? 1 : 0;\n\
\tfor ( ; i < 4 ; i += 2 - includeWidth ) {\n\
\t\twhich = cssExpand[ i ];\n\
\t\tattrs[ \"margin\" + which ] = attrs[ \"padding\" + which ] = type;\n\
\t}\n\
\n\
\tif ( includeWidth ) {\n\
\t\tattrs.opacity = attrs.width = type;\n\
\t}\n\
\n\
\treturn attrs;\n\
}\n\
\n\
function createTween( value, prop, animation ) {\n\
\tvar tween,\n\
\t\tcollection = ( tweeners[ prop ] || [] ).concat( tweeners[ \"*\" ] ),\n\
\t\tindex = 0,\n\
\t\tlength = collection.length;\n\
\tfor ( ; index < length; index++ ) {\n\
\t\tif ( (tween = collection[ index ].call( animation, prop, value )) ) {\n\
\n\
\t\t\t// we're done with this property\n\
\t\t\treturn tween;\n\
\t\t}\n\
\t}\n\
}\n\
\n\
function defaultPrefilter( elem, props, opts ) {\n\
\t/* jshint validthis: true */\n\
\tvar prop, value, toggle, tween, hooks, oldfire, display, checkDisplay,\n\
\t\tanim = this,\n\
\t\torig = {},\n\
\t\tstyle = elem.style,\n\
\t\thidden = elem.nodeType && isHidden( elem ),\n\
\t\tdataShow = jQuery._data( elem, \"fxshow\" );\n\
\n\
\t// handle queue: false promises\n\
\tif ( !opts.queue ) {\n\
\t\thooks = jQuery._queueHooks( elem, \"fx\" );\n\
\t\tif ( hooks.unqueued == null ) {\n\
\t\t\thooks.unqueued = 0;\n\
\t\t\toldfire = hooks.empty.fire;\n\
\t\t\thooks.empty.fire = function() {\n\
\t\t\t\tif ( !hooks.unqueued ) {\n\
\t\t\t\t\toldfire();\n\
\t\t\t\t}\n\
\t\t\t};\n\
\t\t}\n\
\t\thooks.unqueued++;\n\
\n\
\t\tanim.always(function() {\n\
\t\t\t// doing this makes sure that the complete handler will be called\n\
\t\t\t// before this completes\n\
\t\t\tanim.always(function() {\n\
\t\t\t\thooks.unqueued--;\n\
\t\t\t\tif ( !jQuery.queue( elem, \"fx\" ).length ) {\n\
\t\t\t\t\thooks.empty.fire();\n\
\t\t\t\t}\n\
\t\t\t});\n\
\t\t});\n\
\t}\n\
\n\
\t// height/width overflow pass\n\
\tif ( elem.nodeType === 1 && ( \"height\" in props || \"width\" in props ) ) {\n\
\t\t// Make sure that nothing sneaks out\n\
\t\t// Record all 3 overflow attributes because IE does not\n\
\t\t// change the overflow attribute when overflowX and\n\
\t\t// overflowY are set to the same value\n\
\t\topts.overflow = [ style.overflow, style.overflowX, style.overflowY ];\n\
\n\
\t\t// Set display property to inline-block for height/width\n\
\t\t// animations on inline elements that are having width/height animated\n\
\t\tdisplay = jQuery.css( elem, \"display\" );\n\
\n\
\t\t// Test default display if display is currently \"none\"\n\
\t\tcheckDisplay = display === \"none\" ?\n\
\t\t\tjQuery._data( elem, \"olddisplay\" ) || defaultDisplay( elem.nodeName ) : display;\n\
\n\
\t\tif ( checkDisplay === \"inline\" && jQuery.css( elem, \"float\" ) === \"none\" ) {\n\
\n\
\t\t\t// inline-level elements accept inline-block;\n\
\t\t\t// block-level elements need to be inline with layout\n\
\t\t\tif ( !support.inlineBlockNeedsLayout || defaultDisplay( elem.nodeName ) === \"inline\" ) {\n\
\t\t\t\tstyle.display = \"inline-block\";\n\
\t\t\t} else {\n\
\t\t\t\tstyle.zoom = 1;\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\tif ( opts.overflow ) {\n\
\t\tstyle.overflow = \"hidden\";\n\
\t\tif ( !support.shrinkWrapBlocks() ) {\n\
\t\t\tanim.always(function() {\n\
\t\t\t\tstyle.overflow = opts.overflow[ 0 ];\n\
\t\t\t\tstyle.overflowX = opts.overflow[ 1 ];\n\
\t\t\t\tstyle.overflowY = opts.overflow[ 2 ];\n\
\t\t\t});\n\
\t\t}\n\
\t}\n\
\n\
\t// show/hide pass\n\
\tfor ( prop in props ) {\n\
\t\tvalue = props[ prop ];\n\
\t\tif ( rfxtypes.exec( value ) ) {\n\
\t\t\tdelete props[ prop ];\n\
\t\t\ttoggle = toggle || value === \"toggle\";\n\
\t\t\tif ( value === ( hidden ? \"hide\" : \"show\" ) ) {\n\
\n\
\t\t\t\t// If there is dataShow left over from a stopped hide or show and we are going to proceed with show, we should pretend to be hidden\n\
\t\t\t\tif ( value === \"show\" && dataShow && dataShow[ prop ] !== undefined ) {\n\
\t\t\t\t\thidden = true;\n\
\t\t\t\t} else {\n\
\t\t\t\t\tcontinue;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t\torig[ prop ] = dataShow && dataShow[ prop ] || jQuery.style( elem, prop );\n\
\n\
\t\t// Any non-fx value stops us from restoring the original display value\n\
\t\t} else {\n\
\t\t\tdisplay = undefined;\n\
\t\t}\n\
\t}\n\
\n\
\tif ( !jQuery.isEmptyObject( orig ) ) {\n\
\t\tif ( dataShow ) {\n\
\t\t\tif ( \"hidden\" in dataShow ) {\n\
\t\t\t\thidden = dataShow.hidden;\n\
\t\t\t}\n\
\t\t} else {\n\
\t\t\tdataShow = jQuery._data( elem, \"fxshow\", {} );\n\
\t\t}\n\
\n\
\t\t// store state if its toggle - enables .stop().toggle() to \"reverse\"\n\
\t\tif ( toggle ) {\n\
\t\t\tdataShow.hidden = !hidden;\n\
\t\t}\n\
\t\tif ( hidden ) {\n\
\t\t\tjQuery( elem ).show();\n\
\t\t} else {\n\
\t\t\tanim.done(function() {\n\
\t\t\t\tjQuery( elem ).hide();\n\
\t\t\t});\n\
\t\t}\n\
\t\tanim.done(function() {\n\
\t\t\tvar prop;\n\
\t\t\tjQuery._removeData( elem, \"fxshow\" );\n\
\t\t\tfor ( prop in orig ) {\n\
\t\t\t\tjQuery.style( elem, prop, orig[ prop ] );\n\
\t\t\t}\n\
\t\t});\n\
\t\tfor ( prop in orig ) {\n\
\t\t\ttween = createTween( hidden ? dataShow[ prop ] : 0, prop, anim );\n\
\n\
\t\t\tif ( !( prop in dataShow ) ) {\n\
\t\t\t\tdataShow[ prop ] = tween.start;\n\
\t\t\t\tif ( hidden ) {\n\
\t\t\t\t\ttween.end = tween.start;\n\
\t\t\t\t\ttween.start = prop === \"width\" || prop === \"height\" ? 1 : 0;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t// If this is a noop like .hide().hide(), restore an overwritten display value\n\
\t} else if ( (display === \"none\" ? defaultDisplay( elem.nodeName ) : display) === \"inline\" ) {\n\
\t\tstyle.display = display;\n\
\t}\n\
}\n\
\n\
function propFilter( props, specialEasing ) {\n\
\tvar index, name, easing, value, hooks;\n\
\n\
\t// camelCase, specialEasing and expand cssHook pass\n\
\tfor ( index in props ) {\n\
\t\tname = jQuery.camelCase( index );\n\
\t\teasing = specialEasing[ name ];\n\
\t\tvalue = props[ index ];\n\
\t\tif ( jQuery.isArray( value ) ) {\n\
\t\t\teasing = value[ 1 ];\n\
\t\t\tvalue = props[ index ] = value[ 0 ];\n\
\t\t}\n\
\n\
\t\tif ( index !== name ) {\n\
\t\t\tprops[ name ] = value;\n\
\t\t\tdelete props[ index ];\n\
\t\t}\n\
\n\
\t\thooks = jQuery.cssHooks[ name ];\n\
\t\tif ( hooks && \"expand\" in hooks ) {\n\
\t\t\tvalue = hooks.expand( value );\n\
\t\t\tdelete props[ name ];\n\
\n\
\t\t\t// not quite $.extend, this wont overwrite keys already present.\n\
\t\t\t// also - reusing 'index' from above because we have the correct \"name\"\n\
\t\t\tfor ( index in value ) {\n\
\t\t\t\tif ( !( index in props ) ) {\n\
\t\t\t\t\tprops[ index ] = value[ index ];\n\
\t\t\t\t\tspecialEasing[ index ] = easing;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t} else {\n\
\t\t\tspecialEasing[ name ] = easing;\n\
\t\t}\n\
\t}\n\
}\n\
\n\
function Animation( elem, properties, options ) {\n\
\tvar result,\n\
\t\tstopped,\n\
\t\tindex = 0,\n\
\t\tlength = animationPrefilters.length,\n\
\t\tdeferred = jQuery.Deferred().always( function() {\n\
\t\t\t// don't match elem in the :animated selector\n\
\t\t\tdelete tick.elem;\n\
\t\t}),\n\
\t\ttick = function() {\n\
\t\t\tif ( stopped ) {\n\
\t\t\t\treturn false;\n\
\t\t\t}\n\
\t\t\tvar currentTime = fxNow || createFxNow(),\n\
\t\t\t\tremaining = Math.max( 0, animation.startTime + animation.duration - currentTime ),\n\
\t\t\t\t// archaic crash bug won't allow us to use 1 - ( 0.5 || 0 ) (#12497)\n\
\t\t\t\ttemp = remaining / animation.duration || 0,\n\
\t\t\t\tpercent = 1 - temp,\n\
\t\t\t\tindex = 0,\n\
\t\t\t\tlength = animation.tweens.length;\n\
\n\
\t\t\tfor ( ; index < length ; index++ ) {\n\
\t\t\t\tanimation.tweens[ index ].run( percent );\n\
\t\t\t}\n\
\n\
\t\t\tdeferred.notifyWith( elem, [ animation, percent, remaining ]);\n\
\n\
\t\t\tif ( percent < 1 && length ) {\n\
\t\t\t\treturn remaining;\n\
\t\t\t} else {\n\
\t\t\t\tdeferred.resolveWith( elem, [ animation ] );\n\
\t\t\t\treturn false;\n\
\t\t\t}\n\
\t\t},\n\
\t\tanimation = deferred.promise({\n\
\t\t\telem: elem,\n\
\t\t\tprops: jQuery.extend( {}, properties ),\n\
\t\t\topts: jQuery.extend( true, { specialEasing: {} }, options ),\n\
\t\t\toriginalProperties: properties,\n\
\t\t\toriginalOptions: options,\n\
\t\t\tstartTime: fxNow || createFxNow(),\n\
\t\t\tduration: options.duration,\n\
\t\t\ttweens: [],\n\
\t\t\tcreateTween: function( prop, end ) {\n\
\t\t\t\tvar tween = jQuery.Tween( elem, animation.opts, prop, end,\n\
\t\t\t\t\t\tanimation.opts.specialEasing[ prop ] || animation.opts.easing );\n\
\t\t\t\tanimation.tweens.push( tween );\n\
\t\t\t\treturn tween;\n\
\t\t\t},\n\
\t\t\tstop: function( gotoEnd ) {\n\
\t\t\t\tvar index = 0,\n\
\t\t\t\t\t// if we are going to the end, we want to run all the tweens\n\
\t\t\t\t\t// otherwise we skip this part\n\
\t\t\t\t\tlength = gotoEnd ? animation.tweens.length : 0;\n\
\t\t\t\tif ( stopped ) {\n\
\t\t\t\t\treturn this;\n\
\t\t\t\t}\n\
\t\t\t\tstopped = true;\n\
\t\t\t\tfor ( ; index < length ; index++ ) {\n\
\t\t\t\t\tanimation.tweens[ index ].run( 1 );\n\
\t\t\t\t}\n\
\n\
\t\t\t\t// resolve when we played the last frame\n\
\t\t\t\t// otherwise, reject\n\
\t\t\t\tif ( gotoEnd ) {\n\
\t\t\t\t\tdeferred.resolveWith( elem, [ animation, gotoEnd ] );\n\
\t\t\t\t} else {\n\
\t\t\t\t\tdeferred.rejectWith( elem, [ animation, gotoEnd ] );\n\
\t\t\t\t}\n\
\t\t\t\treturn this;\n\
\t\t\t}\n\
\t\t}),\n\
\t\tprops = animation.props;\n\
\n\
\tpropFilter( props, animation.opts.specialEasing );\n\
\n\
\tfor ( ; index < length ; index++ ) {\n\
\t\tresult = animationPrefilters[ index ].call( animation, elem, props, animation.opts );\n\
\t\tif ( result ) {\n\
\t\t\treturn result;\n\
\t\t}\n\
\t}\n\
\n\
\tjQuery.map( props, createTween, animation );\n\
\n\
\tif ( jQuery.isFunction( animation.opts.start ) ) {\n\
\t\tanimation.opts.start.call( elem, animation );\n\
\t}\n\
\n\
\tjQuery.fx.timer(\n\
\t\tjQuery.extend( tick, {\n\
\t\t\telem: elem,\n\
\t\t\tanim: animation,\n\
\t\t\tqueue: animation.opts.queue\n\
\t\t})\n\
\t);\n\
\n\
\t// attach callbacks from options\n\
\treturn animation.progress( animation.opts.progress )\n\
\t\t.done( animation.opts.done, animation.opts.complete )\n\
\t\t.fail( animation.opts.fail )\n\
\t\t.always( animation.opts.always );\n\
}\n\
\n\
jQuery.Animation = jQuery.extend( Animation, {\n\
\ttweener: function( props, callback ) {\n\
\t\tif ( jQuery.isFunction( props ) ) {\n\
\t\t\tcallback = props;\n\
\t\t\tprops = [ \"*\" ];\n\
\t\t} else {\n\
\t\t\tprops = props.split(\" \");\n\
\t\t}\n\
\n\
\t\tvar prop,\n\
\t\t\tindex = 0,\n\
\t\t\tlength = props.length;\n\
\n\
\t\tfor ( ; index < length ; index++ ) {\n\
\t\t\tprop = props[ index ];\n\
\t\t\ttweeners[ prop ] = tweeners[ prop ] || [];\n\
\t\t\ttweeners[ prop ].unshift( callback );\n\
\t\t}\n\
\t},\n\
\n\
\tprefilter: function( callback, prepend ) {\n\
\t\tif ( prepend ) {\n\
\t\t\tanimationPrefilters.unshift( callback );\n\
\t\t} else {\n\
\t\t\tanimationPrefilters.push( callback );\n\
\t\t}\n\
\t}\n\
});\n\
\n\
jQuery.speed = function( speed, easing, fn ) {\n\
\tvar opt = speed && typeof speed === \"object\" ? jQuery.extend( {}, speed ) : {\n\
\t\tcomplete: fn || !fn && easing ||\n\
\t\t\tjQuery.isFunction( speed ) && speed,\n\
\t\tduration: speed,\n\
\t\teasing: fn && easing || easing && !jQuery.isFunction( easing ) && easing\n\
\t};\n\
\n\
\topt.duration = jQuery.fx.off ? 0 : typeof opt.duration === \"number\" ? opt.duration :\n\
\t\topt.duration in jQuery.fx.speeds ? jQuery.fx.speeds[ opt.duration ] : jQuery.fx.speeds._default;\n\
\n\
\t// normalize opt.queue - true/undefined/null -> \"fx\"\n\
\tif ( opt.queue == null || opt.queue === true ) {\n\
\t\topt.queue = \"fx\";\n\
\t}\n\
\n\
\t// Queueing\n\
\topt.old = opt.complete;\n\
\n\
\topt.complete = function() {\n\
\t\tif ( jQuery.isFunction( opt.old ) ) {\n\
\t\t\topt.old.call( this );\n\
\t\t}\n\
\n\
\t\tif ( opt.queue ) {\n\
\t\t\tjQuery.dequeue( this, opt.queue );\n\
\t\t}\n\
\t};\n\
\n\
\treturn opt;\n\
};\n\
\n\
jQuery.fn.extend({\n\
\tfadeTo: function( speed, to, easing, callback ) {\n\
\n\
\t\t// show any hidden elements after setting opacity to 0\n\
\t\treturn this.filter( isHidden ).css( \"opacity\", 0 ).show()\n\
\n\
\t\t\t// animate to the value specified\n\
\t\t\t.end().animate({ opacity: to }, speed, easing, callback );\n\
\t},\n\
\tanimate: function( prop, speed, easing, callback ) {\n\
\t\tvar empty = jQuery.isEmptyObject( prop ),\n\
\t\t\toptall = jQuery.speed( speed, easing, callback ),\n\
\t\t\tdoAnimation = function() {\n\
\t\t\t\t// Operate on a copy of prop so per-property easing won't be lost\n\
\t\t\t\tvar anim = Animation( this, jQuery.extend( {}, prop ), optall );\n\
\n\
\t\t\t\t// Empty animations, or finishing resolves immediately\n\
\t\t\t\tif ( empty || jQuery._data( this, \"finish\" ) ) {\n\
\t\t\t\t\tanim.stop( true );\n\
\t\t\t\t}\n\
\t\t\t};\n\
\t\t\tdoAnimation.finish = doAnimation;\n\
\n\
\t\treturn empty || optall.queue === false ?\n\
\t\t\tthis.each( doAnimation ) :\n\
\t\t\tthis.queue( optall.queue, doAnimation );\n\
\t},\n\
\tstop: function( type, clearQueue, gotoEnd ) {\n\
\t\tvar stopQueue = function( hooks ) {\n\
\t\t\tvar stop = hooks.stop;\n\
\t\t\tdelete hooks.stop;\n\
\t\t\tstop( gotoEnd );\n\
\t\t};\n\
\n\
\t\tif ( typeof type !== \"string\" ) {\n\
\t\t\tgotoEnd = clearQueue;\n\
\t\t\tclearQueue = type;\n\
\t\t\ttype = undefined;\n\
\t\t}\n\
\t\tif ( clearQueue && type !== false ) {\n\
\t\t\tthis.queue( type || \"fx\", [] );\n\
\t\t}\n\
\n\
\t\treturn this.each(function() {\n\
\t\t\tvar dequeue = true,\n\
\t\t\t\tindex = type != null && type + \"queueHooks\",\n\
\t\t\t\ttimers = jQuery.timers,\n\
\t\t\t\tdata = jQuery._data( this );\n\
\n\
\t\t\tif ( index ) {\n\
\t\t\t\tif ( data[ index ] && data[ index ].stop ) {\n\
\t\t\t\t\tstopQueue( data[ index ] );\n\
\t\t\t\t}\n\
\t\t\t} else {\n\
\t\t\t\tfor ( index in data ) {\n\
\t\t\t\t\tif ( data[ index ] && data[ index ].stop && rrun.test( index ) ) {\n\
\t\t\t\t\t\tstopQueue( data[ index ] );\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t\tfor ( index = timers.length; index--; ) {\n\
\t\t\t\tif ( timers[ index ].elem === this && (type == null || timers[ index ].queue === type) ) {\n\
\t\t\t\t\ttimers[ index ].anim.stop( gotoEnd );\n\
\t\t\t\t\tdequeue = false;\n\
\t\t\t\t\ttimers.splice( index, 1 );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t\t// start the next in the queue if the last step wasn't forced\n\
\t\t\t// timers currently will call their complete callbacks, which will dequeue\n\
\t\t\t// but only if they were gotoEnd\n\
\t\t\tif ( dequeue || !gotoEnd ) {\n\
\t\t\t\tjQuery.dequeue( this, type );\n\
\t\t\t}\n\
\t\t});\n\
\t},\n\
\tfinish: function( type ) {\n\
\t\tif ( type !== false ) {\n\
\t\t\ttype = type || \"fx\";\n\
\t\t}\n\
\t\treturn this.each(function() {\n\
\t\t\tvar index,\n\
\t\t\t\tdata = jQuery._data( this ),\n\
\t\t\t\tqueue = data[ type + \"queue\" ],\n\
\t\t\t\thooks = data[ type + \"queueHooks\" ],\n\
\t\t\t\ttimers = jQuery.timers,\n\
\t\t\t\tlength = queue ? queue.length : 0;\n\
\n\
\t\t\t// enable finishing flag on private data\n\
\t\t\tdata.finish = true;\n\
\n\
\t\t\t// empty the queue first\n\
\t\t\tjQuery.queue( this, type, [] );\n\
\n\
\t\t\tif ( hooks && hooks.stop ) {\n\
\t\t\t\thooks.stop.call( this, true );\n\
\t\t\t}\n\
\n\
\t\t\t// look for any active animations, and finish them\n\
\t\t\tfor ( index = timers.length; index--; ) {\n\
\t\t\t\tif ( timers[ index ].elem === this && timers[ index ].queue === type ) {\n\
\t\t\t\t\ttimers[ index ].anim.stop( true );\n\
\t\t\t\t\ttimers.splice( index, 1 );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t\t// look for any animations in the old queue and finish them\n\
\t\t\tfor ( index = 0; index < length; index++ ) {\n\
\t\t\t\tif ( queue[ index ] && queue[ index ].finish ) {\n\
\t\t\t\t\tqueue[ index ].finish.call( this );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t\t// turn off finishing flag\n\
\t\t\tdelete data.finish;\n\
\t\t});\n\
\t}\n\
});\n\
\n\
jQuery.each([ \"toggle\", \"show\", \"hide\" ], function( i, name ) {\n\
\tvar cssFn = jQuery.fn[ name ];\n\
\tjQuery.fn[ name ] = function( speed, easing, callback ) {\n\
\t\treturn speed == null || typeof speed === \"boolean\" ?\n\
\t\t\tcssFn.apply( this, arguments ) :\n\
\t\t\tthis.animate( genFx( name, true ), speed, easing, callback );\n\
\t};\n\
});\n\
\n\
// Generate shortcuts for custom animations\n\
jQuery.each({\n\
\tslideDown: genFx(\"show\"),\n\
\tslideUp: genFx(\"hide\"),\n\
\tslideToggle: genFx(\"toggle\"),\n\
\tfadeIn: { opacity: \"show\" },\n\
\tfadeOut: { opacity: \"hide\" },\n\
\tfadeToggle: { opacity: \"toggle\" }\n\
}, function( name, props ) {\n\
\tjQuery.fn[ name ] = function( speed, easing, callback ) {\n\
\t\treturn this.animate( props, speed, easing, callback );\n\
\t};\n\
});\n\
\n\
jQuery.timers = [];\n\
jQuery.fx.tick = function() {\n\
\tvar timer,\n\
\t\ttimers = jQuery.timers,\n\
\t\ti = 0;\n\
\n\
\tfxNow = jQuery.now();\n\
\n\
\tfor ( ; i < timers.length; i++ ) {\n\
\t\ttimer = timers[ i ];\n\
\t\t// Checks the timer has not already been removed\n\
\t\tif ( !timer() && timers[ i ] === timer ) {\n\
\t\t\ttimers.splice( i--, 1 );\n\
\t\t}\n\
\t}\n\
\n\
\tif ( !timers.length ) {\n\
\t\tjQuery.fx.stop();\n\
\t}\n\
\tfxNow = undefined;\n\
};\n\
\n\
jQuery.fx.timer = function( timer ) {\n\
\tjQuery.timers.push( timer );\n\
\tif ( timer() ) {\n\
\t\tjQuery.fx.start();\n\
\t} else {\n\
\t\tjQuery.timers.pop();\n\
\t}\n\
};\n\
\n\
jQuery.fx.interval = 13;\n\
\n\
jQuery.fx.start = function() {\n\
\tif ( !timerId ) {\n\
\t\ttimerId = setInterval( jQuery.fx.tick, jQuery.fx.interval );\n\
\t}\n\
};\n\
\n\
jQuery.fx.stop = function() {\n\
\tclearInterval( timerId );\n\
\ttimerId = null;\n\
};\n\
\n\
jQuery.fx.speeds = {\n\
\tslow: 600,\n\
\tfast: 200,\n\
\t// Default speed\n\
\t_default: 400\n\
};\n\
\n\
\n\
// Based off of the plugin by Clint Helfers, with permission.\n\
// http://blindsignals.com/index.php/2009/07/jquery-delay/\n\
jQuery.fn.delay = function( time, type ) {\n\
\ttime = jQuery.fx ? jQuery.fx.speeds[ time ] || time : time;\n\
\ttype = type || \"fx\";\n\
\n\
\treturn this.queue( type, function( next, hooks ) {\n\
\t\tvar timeout = setTimeout( next, time );\n\
\t\thooks.stop = function() {\n\
\t\t\tclearTimeout( timeout );\n\
\t\t};\n\
\t});\n\
};\n\
\n\
\n\
(function() {\n\
\t// Minified: var a,b,c,d,e\n\
\tvar input, div, select, a, opt;\n\
\n\
\t// Setup\n\
\tdiv = document.createElement( \"div\" );\n\
\tdiv.setAttribute( \"className\", \"t\" );\n\
\tdiv.innerHTML = \"  <link/><table></table><a href='/a'>a</a><input type='checkbox'/>\";\n\
\ta = div.getElementsByTagName(\"a\")[ 0 ];\n\
\n\
\t// First batch of tests.\n\
\tselect = document.createElement(\"select\");\n\
\topt = select.appendChild( document.createElement(\"option\") );\n\
\tinput = div.getElementsByTagName(\"input\")[ 0 ];\n\
\n\
\ta.style.cssText = \"top:1px\";\n\
\n\
\t// Test setAttribute on camelCase class. If it works, we need attrFixes when doing get/setAttribute (ie6/7)\n\
\tsupport.getSetAttribute = div.className !== \"t\";\n\
\n\
\t// Get the style information from getAttribute\n\
\t// (IE uses .cssText instead)\n\
\tsupport.style = /top/.test( a.getAttribute(\"style\") );\n\
\n\
\t// Make sure that URLs aren't manipulated\n\
\t// (IE normalizes it by default)\n\
\tsupport.hrefNormalized = a.getAttribute(\"href\") === \"/a\";\n\
\n\
\t// Check the default checkbox/radio value (\"\" on WebKit; \"on\" elsewhere)\n\
\tsupport.checkOn = !!input.value;\n\
\n\
\t// Make sure that a selected-by-default option has a working selected property.\n\
\t// (WebKit defaults to false instead of true, IE too, if it's in an optgroup)\n\
\tsupport.optSelected = opt.selected;\n\
\n\
\t// Tests for enctype support on a form (#6743)\n\
\tsupport.enctype = !!document.createElement(\"form\").enctype;\n\
\n\
\t// Make sure that the options inside disabled selects aren't marked as disabled\n\
\t// (WebKit marks them as disabled)\n\
\tselect.disabled = true;\n\
\tsupport.optDisabled = !opt.disabled;\n\
\n\
\t// Support: IE8 only\n\
\t// Check if we can trust getAttribute(\"value\")\n\
\tinput = document.createElement( \"input\" );\n\
\tinput.setAttribute( \"value\", \"\" );\n\
\tsupport.input = input.getAttribute( \"value\" ) === \"\";\n\
\n\
\t// Check if an input maintains its value after becoming a radio\n\
\tinput.value = \"t\";\n\
\tinput.setAttribute( \"type\", \"radio\" );\n\
\tsupport.radioValue = input.value === \"t\";\n\
})();\n\
\n\
\n\
var rreturn = /\\r/g;\n\
\n\
jQuery.fn.extend({\n\
\tval: function( value ) {\n\
\t\tvar hooks, ret, isFunction,\n\
\t\t\telem = this[0];\n\
\n\
\t\tif ( !arguments.length ) {\n\
\t\t\tif ( elem ) {\n\
\t\t\t\thooks = jQuery.valHooks[ elem.type ] || jQuery.valHooks[ elem.nodeName.toLowerCase() ];\n\
\n\
\t\t\t\tif ( hooks && \"get\" in hooks && (ret = hooks.get( elem, \"value\" )) !== undefined ) {\n\
\t\t\t\t\treturn ret;\n\
\t\t\t\t}\n\
\n\
\t\t\t\tret = elem.value;\n\
\n\
\t\t\t\treturn typeof ret === \"string\" ?\n\
\t\t\t\t\t// handle most common string cases\n\
\t\t\t\t\tret.replace(rreturn, \"\") :\n\
\t\t\t\t\t// handle cases where value is null/undef or number\n\
\t\t\t\t\tret == null ? \"\" : ret;\n\
\t\t\t}\n\
\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\tisFunction = jQuery.isFunction( value );\n\
\n\
\t\treturn this.each(function( i ) {\n\
\t\t\tvar val;\n\
\n\
\t\t\tif ( this.nodeType !== 1 ) {\n\
\t\t\t\treturn;\n\
\t\t\t}\n\
\n\
\t\t\tif ( isFunction ) {\n\
\t\t\t\tval = value.call( this, i, jQuery( this ).val() );\n\
\t\t\t} else {\n\
\t\t\t\tval = value;\n\
\t\t\t}\n\
\n\
\t\t\t// Treat null/undefined as \"\"; convert numbers to string\n\
\t\t\tif ( val == null ) {\n\
\t\t\t\tval = \"\";\n\
\t\t\t} else if ( typeof val === \"number\" ) {\n\
\t\t\t\tval += \"\";\n\
\t\t\t} else if ( jQuery.isArray( val ) ) {\n\
\t\t\t\tval = jQuery.map( val, function( value ) {\n\
\t\t\t\t\treturn value == null ? \"\" : value + \"\";\n\
\t\t\t\t});\n\
\t\t\t}\n\
\n\
\t\t\thooks = jQuery.valHooks[ this.type ] || jQuery.valHooks[ this.nodeName.toLowerCase() ];\n\
\n\
\t\t\t// If set returns undefined, fall back to normal setting\n\
\t\t\tif ( !hooks || !(\"set\" in hooks) || hooks.set( this, val, \"value\" ) === undefined ) {\n\
\t\t\t\tthis.value = val;\n\
\t\t\t}\n\
\t\t});\n\
\t}\n\
});\n\
\n\
jQuery.extend({\n\
\tvalHooks: {\n\
\t\toption: {\n\
\t\t\tget: function( elem ) {\n\
\t\t\t\tvar val = jQuery.find.attr( elem, \"value\" );\n\
\t\t\t\treturn val != null ?\n\
\t\t\t\t\tval :\n\
\t\t\t\t\t// Support: IE10-11+\n\
\t\t\t\t\t// option.text throws exceptions (#14686, #14858)\n\
\t\t\t\t\tjQuery.trim( jQuery.text( elem ) );\n\
\t\t\t}\n\
\t\t},\n\
\t\tselect: {\n\
\t\t\tget: function( elem ) {\n\
\t\t\t\tvar value, option,\n\
\t\t\t\t\toptions = elem.options,\n\
\t\t\t\t\tindex = elem.selectedIndex,\n\
\t\t\t\t\tone = elem.type === \"select-one\" || index < 0,\n\
\t\t\t\t\tvalues = one ? null : [],\n\
\t\t\t\t\tmax = one ? index + 1 : options.length,\n\
\t\t\t\t\ti = index < 0 ?\n\
\t\t\t\t\t\tmax :\n\
\t\t\t\t\t\tone ? index : 0;\n\
\n\
\t\t\t\t// Loop through all the selected options\n\
\t\t\t\tfor ( ; i < max; i++ ) {\n\
\t\t\t\t\toption = options[ i ];\n\
\n\
\t\t\t\t\t// oldIE doesn't update selected after form reset (#2551)\n\
\t\t\t\t\tif ( ( option.selected || i === index ) &&\n\
\t\t\t\t\t\t\t// Don't return options that are disabled or in a disabled optgroup\n\
\t\t\t\t\t\t\t( support.optDisabled ? !option.disabled : option.getAttribute(\"disabled\") === null ) &&\n\
\t\t\t\t\t\t\t( !option.parentNode.disabled || !jQuery.nodeName( option.parentNode, \"optgroup\" ) ) ) {\n\
\n\
\t\t\t\t\t\t// Get the specific value for the option\n\
\t\t\t\t\t\tvalue = jQuery( option ).val();\n\
\n\
\t\t\t\t\t\t// We don't need an array for one selects\n\
\t\t\t\t\t\tif ( one ) {\n\
\t\t\t\t\t\t\treturn value;\n\
\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t\t// Multi-Selects return an array\n\
\t\t\t\t\t\tvalues.push( value );\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\n\
\t\t\t\treturn values;\n\
\t\t\t},\n\
\n\
\t\t\tset: function( elem, value ) {\n\
\t\t\t\tvar optionSet, option,\n\
\t\t\t\t\toptions = elem.options,\n\
\t\t\t\t\tvalues = jQuery.makeArray( value ),\n\
\t\t\t\t\ti = options.length;\n\
\n\
\t\t\t\twhile ( i-- ) {\n\
\t\t\t\t\toption = options[ i ];\n\
\n\
\t\t\t\t\tif ( jQuery.inArray( jQuery.valHooks.option.get( option ), values ) >= 0 ) {\n\
\n\
\t\t\t\t\t\t// Support: IE6\n\
\t\t\t\t\t\t// When new option element is added to select box we need to\n\
\t\t\t\t\t\t// force reflow of newly added node in order to workaround delay\n\
\t\t\t\t\t\t// of initialization properties\n\
\t\t\t\t\t\ttry {\n\
\t\t\t\t\t\t\toption.selected = optionSet = true;\n\
\n\
\t\t\t\t\t\t} catch ( _ ) {\n\
\n\
\t\t\t\t\t\t\t// Will be executed only in IE6\n\
\t\t\t\t\t\t\toption.scrollHeight;\n\
\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t} else {\n\
\t\t\t\t\t\toption.selected = false;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\n\
\t\t\t\t// Force browsers to behave consistently when non-matching value is set\n\
\t\t\t\tif ( !optionSet ) {\n\
\t\t\t\t\telem.selectedIndex = -1;\n\
\t\t\t\t}\n\
\n\
\t\t\t\treturn options;\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
});\n\
\n\
// Radios and checkboxes getter/setter\n\
jQuery.each([ \"radio\", \"checkbox\" ], function() {\n\
\tjQuery.valHooks[ this ] = {\n\
\t\tset: function( elem, value ) {\n\
\t\t\tif ( jQuery.isArray( value ) ) {\n\
\t\t\t\treturn ( elem.checked = jQuery.inArray( jQuery(elem).val(), value ) >= 0 );\n\
\t\t\t}\n\
\t\t}\n\
\t};\n\
\tif ( !support.checkOn ) {\n\
\t\tjQuery.valHooks[ this ].get = function( elem ) {\n\
\t\t\t// Support: Webkit\n\
\t\t\t// \"\" is returned instead of \"on\" if a value isn't specified\n\
\t\t\treturn elem.getAttribute(\"value\") === null ? \"on\" : elem.value;\n\
\t\t};\n\
\t}\n\
});\n\
\n\
\n\
\n\
\n\
var nodeHook, boolHook,\n\
\tattrHandle = jQuery.expr.attrHandle,\n\
\truseDefault = /^(?:checked|selected)$/i,\n\
\tgetSetAttribute = support.getSetAttribute,\n\
\tgetSetInput = support.input;\n\
\n\
jQuery.fn.extend({\n\
\tattr: function( name, value ) {\n\
\t\treturn access( this, jQuery.attr, name, value, arguments.length > 1 );\n\
\t},\n\
\n\
\tremoveAttr: function( name ) {\n\
\t\treturn this.each(function() {\n\
\t\t\tjQuery.removeAttr( this, name );\n\
\t\t});\n\
\t}\n\
});\n\
\n\
jQuery.extend({\n\
\tattr: function( elem, name, value ) {\n\
\t\tvar hooks, ret,\n\
\t\t\tnType = elem.nodeType;\n\
\n\
\t\t// don't get/set attributes on text, comment and attribute nodes\n\
\t\tif ( !elem || nType === 3 || nType === 8 || nType === 2 ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\t// Fallback to prop when attributes are not supported\n\
\t\tif ( typeof elem.getAttribute === strundefined ) {\n\
\t\t\treturn jQuery.prop( elem, name, value );\n\
\t\t}\n\
\n\
\t\t// All attributes are lowercase\n\
\t\t// Grab necessary hook if one is defined\n\
\t\tif ( nType !== 1 || !jQuery.isXMLDoc( elem ) ) {\n\
\t\t\tname = name.toLowerCase();\n\
\t\t\thooks = jQuery.attrHooks[ name ] ||\n\
\t\t\t\t( jQuery.expr.match.bool.test( name ) ? boolHook : nodeHook );\n\
\t\t}\n\
\n\
\t\tif ( value !== undefined ) {\n\
\n\
\t\t\tif ( value === null ) {\n\
\t\t\t\tjQuery.removeAttr( elem, name );\n\
\n\
\t\t\t} else if ( hooks && \"set\" in hooks && (ret = hooks.set( elem, value, name )) !== undefined ) {\n\
\t\t\t\treturn ret;\n\
\n\
\t\t\t} else {\n\
\t\t\t\telem.setAttribute( name, value + \"\" );\n\
\t\t\t\treturn value;\n\
\t\t\t}\n\
\n\
\t\t} else if ( hooks && \"get\" in hooks && (ret = hooks.get( elem, name )) !== null ) {\n\
\t\t\treturn ret;\n\
\n\
\t\t} else {\n\
\t\t\tret = jQuery.find.attr( elem, name );\n\
\n\
\t\t\t// Non-existent attributes return null, we normalize to undefined\n\
\t\t\treturn ret == null ?\n\
\t\t\t\tundefined :\n\
\t\t\t\tret;\n\
\t\t}\n\
\t},\n\
\n\
\tremoveAttr: function( elem, value ) {\n\
\t\tvar name, propName,\n\
\t\t\ti = 0,\n\
\t\t\tattrNames = value && value.match( rnotwhite );\n\
\n\
\t\tif ( attrNames && elem.nodeType === 1 ) {\n\
\t\t\twhile ( (name = attrNames[i++]) ) {\n\
\t\t\t\tpropName = jQuery.propFix[ name ] || name;\n\
\n\
\t\t\t\t// Boolean attributes get special treatment (#10870)\n\
\t\t\t\tif ( jQuery.expr.match.bool.test( name ) ) {\n\
\t\t\t\t\t// Set corresponding property to false\n\
\t\t\t\t\tif ( getSetInput && getSetAttribute || !ruseDefault.test( name ) ) {\n\
\t\t\t\t\t\telem[ propName ] = false;\n\
\t\t\t\t\t// Support: IE<9\n\
\t\t\t\t\t// Also clear defaultChecked/defaultSelected (if appropriate)\n\
\t\t\t\t\t} else {\n\
\t\t\t\t\t\telem[ jQuery.camelCase( \"default-\" + name ) ] =\n\
\t\t\t\t\t\t\telem[ propName ] = false;\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t// See #9699 for explanation of this approach (setting first, then removal)\n\
\t\t\t\t} else {\n\
\t\t\t\t\tjQuery.attr( elem, name, \"\" );\n\
\t\t\t\t}\n\
\n\
\t\t\t\telem.removeAttribute( getSetAttribute ? name : propName );\n\
\t\t\t}\n\
\t\t}\n\
\t},\n\
\n\
\tattrHooks: {\n\
\t\ttype: {\n\
\t\t\tset: function( elem, value ) {\n\
\t\t\t\tif ( !support.radioValue && value === \"radio\" && jQuery.nodeName(elem, \"input\") ) {\n\
\t\t\t\t\t// Setting the type on a radio button after the value resets the value in IE6-9\n\
\t\t\t\t\t// Reset value to default in case type is set after value during creation\n\
\t\t\t\t\tvar val = elem.value;\n\
\t\t\t\t\telem.setAttribute( \"type\", value );\n\
\t\t\t\t\tif ( val ) {\n\
\t\t\t\t\t\telem.value = val;\n\
\t\t\t\t\t}\n\
\t\t\t\t\treturn value;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
});\n\
\n\
// Hook for boolean attributes\n\
boolHook = {\n\
\tset: function( elem, value, name ) {\n\
\t\tif ( value === false ) {\n\
\t\t\t// Remove boolean attributes when set to false\n\
\t\t\tjQuery.removeAttr( elem, name );\n\
\t\t} else if ( getSetInput && getSetAttribute || !ruseDefault.test( name ) ) {\n\
\t\t\t// IE<8 needs the *property* name\n\
\t\t\telem.setAttribute( !getSetAttribute && jQuery.propFix[ name ] || name, name );\n\
\n\
\t\t// Use defaultChecked and defaultSelected for oldIE\n\
\t\t} else {\n\
\t\t\telem[ jQuery.camelCase( \"default-\" + name ) ] = elem[ name ] = true;\n\
\t\t}\n\
\n\
\t\treturn name;\n\
\t}\n\
};\n\
\n\
// Retrieve booleans specially\n\
jQuery.each( jQuery.expr.match.bool.source.match( /\\w+/g ), function( i, name ) {\n\
\n\
\tvar getter = attrHandle[ name ] || jQuery.find.attr;\n\
\n\
\tattrHandle[ name ] = getSetInput && getSetAttribute || !ruseDefault.test( name ) ?\n\
\t\tfunction( elem, name, isXML ) {\n\
\t\t\tvar ret, handle;\n\
\t\t\tif ( !isXML ) {\n\
\t\t\t\t// Avoid an infinite loop by temporarily removing this function from the getter\n\
\t\t\t\thandle = attrHandle[ name ];\n\
\t\t\t\tattrHandle[ name ] = ret;\n\
\t\t\t\tret = getter( elem, name, isXML ) != null ?\n\
\t\t\t\t\tname.toLowerCase() :\n\
\t\t\t\t\tnull;\n\
\t\t\t\tattrHandle[ name ] = handle;\n\
\t\t\t}\n\
\t\t\treturn ret;\n\
\t\t} :\n\
\t\tfunction( elem, name, isXML ) {\n\
\t\t\tif ( !isXML ) {\n\
\t\t\t\treturn elem[ jQuery.camelCase( \"default-\" + name ) ] ?\n\
\t\t\t\t\tname.toLowerCase() :\n\
\t\t\t\t\tnull;\n\
\t\t\t}\n\
\t\t};\n\
});\n\
\n\
// fix oldIE attroperties\n\
if ( !getSetInput || !getSetAttribute ) {\n\
\tjQuery.attrHooks.value = {\n\
\t\tset: function( elem, value, name ) {\n\
\t\t\tif ( jQuery.nodeName( elem, \"input\" ) ) {\n\
\t\t\t\t// Does not return so that setAttribute is also used\n\
\t\t\t\telem.defaultValue = value;\n\
\t\t\t} else {\n\
\t\t\t\t// Use nodeHook if defined (#1954); otherwise setAttribute is fine\n\
\t\t\t\treturn nodeHook && nodeHook.set( elem, value, name );\n\
\t\t\t}\n\
\t\t}\n\
\t};\n\
}\n\
\n\
// IE6/7 do not support getting/setting some attributes with get/setAttribute\n\
if ( !getSetAttribute ) {\n\
\n\
\t// Use this for any attribute in IE6/7\n\
\t// This fixes almost every IE6/7 issue\n\
\tnodeHook = {\n\
\t\tset: function( elem, value, name ) {\n\
\t\t\t// Set the existing or create a new attribute node\n\
\t\t\tvar ret = elem.getAttributeNode( name );\n\
\t\t\tif ( !ret ) {\n\
\t\t\t\telem.setAttributeNode(\n\
\t\t\t\t\t(ret = elem.ownerDocument.createAttribute( name ))\n\
\t\t\t\t);\n\
\t\t\t}\n\
\n\
\t\t\tret.value = value += \"\";\n\
\n\
\t\t\t// Break association with cloned elements by also using setAttribute (#9646)\n\
\t\t\tif ( name === \"value\" || value === elem.getAttribute( name ) ) {\n\
\t\t\t\treturn value;\n\
\t\t\t}\n\
\t\t}\n\
\t};\n\
\n\
\t// Some attributes are constructed with empty-string values when not defined\n\
\tattrHandle.id = attrHandle.name = attrHandle.coords =\n\
\t\tfunction( elem, name, isXML ) {\n\
\t\t\tvar ret;\n\
\t\t\tif ( !isXML ) {\n\
\t\t\t\treturn (ret = elem.getAttributeNode( name )) && ret.value !== \"\" ?\n\
\t\t\t\t\tret.value :\n\
\t\t\t\t\tnull;\n\
\t\t\t}\n\
\t\t};\n\
\n\
\t// Fixing value retrieval on a button requires this module\n\
\tjQuery.valHooks.button = {\n\
\t\tget: function( elem, name ) {\n\
\t\t\tvar ret = elem.getAttributeNode( name );\n\
\t\t\tif ( ret && ret.specified ) {\n\
\t\t\t\treturn ret.value;\n\
\t\t\t}\n\
\t\t},\n\
\t\tset: nodeHook.set\n\
\t};\n\
\n\
\t// Set contenteditable to false on removals(#10429)\n\
\t// Setting to empty string throws an error as an invalid value\n\
\tjQuery.attrHooks.contenteditable = {\n\
\t\tset: function( elem, value, name ) {\n\
\t\t\tnodeHook.set( elem, value === \"\" ? false : value, name );\n\
\t\t}\n\
\t};\n\
\n\
\t// Set width and height to auto instead of 0 on empty string( Bug #8150 )\n\
\t// This is for removals\n\
\tjQuery.each([ \"width\", \"height\" ], function( i, name ) {\n\
\t\tjQuery.attrHooks[ name ] = {\n\
\t\t\tset: function( elem, value ) {\n\
\t\t\t\tif ( value === \"\" ) {\n\
\t\t\t\t\telem.setAttribute( name, \"auto\" );\n\
\t\t\t\t\treturn value;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t};\n\
\t});\n\
}\n\
\n\
if ( !support.style ) {\n\
\tjQuery.attrHooks.style = {\n\
\t\tget: function( elem ) {\n\
\t\t\t// Return undefined in the case of empty string\n\
\t\t\t// Note: IE uppercases css property names, but if we were to .toLowerCase()\n\
\t\t\t// .cssText, that would destroy case senstitivity in URL's, like in \"background\"\n\
\t\t\treturn elem.style.cssText || undefined;\n\
\t\t},\n\
\t\tset: function( elem, value ) {\n\
\t\t\treturn ( elem.style.cssText = value + \"\" );\n\
\t\t}\n\
\t};\n\
}\n\
\n\
\n\
\n\
\n\
var rfocusable = /^(?:input|select|textarea|button|object)$/i,\n\
\trclickable = /^(?:a|area)$/i;\n\
\n\
jQuery.fn.extend({\n\
\tprop: function( name, value ) {\n\
\t\treturn access( this, jQuery.prop, name, value, arguments.length > 1 );\n\
\t},\n\
\n\
\tremoveProp: function( name ) {\n\
\t\tname = jQuery.propFix[ name ] || name;\n\
\t\treturn this.each(function() {\n\
\t\t\t// try/catch handles cases where IE balks (such as removing a property on window)\n\
\t\t\ttry {\n\
\t\t\t\tthis[ name ] = undefined;\n\
\t\t\t\tdelete this[ name ];\n\
\t\t\t} catch( e ) {}\n\
\t\t});\n\
\t}\n\
});\n\
\n\
jQuery.extend({\n\
\tpropFix: {\n\
\t\t\"for\": \"htmlFor\",\n\
\t\t\"class\": \"className\"\n\
\t},\n\
\n\
\tprop: function( elem, name, value ) {\n\
\t\tvar ret, hooks, notxml,\n\
\t\t\tnType = elem.nodeType;\n\
\n\
\t\t// don't get/set properties on text, comment and attribute nodes\n\
\t\tif ( !elem || nType === 3 || nType === 8 || nType === 2 ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\tnotxml = nType !== 1 || !jQuery.isXMLDoc( elem );\n\
\n\
\t\tif ( notxml ) {\n\
\t\t\t// Fix name and attach hooks\n\
\t\t\tname = jQuery.propFix[ name ] || name;\n\
\t\t\thooks = jQuery.propHooks[ name ];\n\
\t\t}\n\
\n\
\t\tif ( value !== undefined ) {\n\
\t\t\treturn hooks && \"set\" in hooks && (ret = hooks.set( elem, value, name )) !== undefined ?\n\
\t\t\t\tret :\n\
\t\t\t\t( elem[ name ] = value );\n\
\n\
\t\t} else {\n\
\t\t\treturn hooks && \"get\" in hooks && (ret = hooks.get( elem, name )) !== null ?\n\
\t\t\t\tret :\n\
\t\t\t\telem[ name ];\n\
\t\t}\n\
\t},\n\
\n\
\tpropHooks: {\n\
\t\ttabIndex: {\n\
\t\t\tget: function( elem ) {\n\
\t\t\t\t// elem.tabIndex doesn't always return the correct value when it hasn't been explicitly set\n\
\t\t\t\t// http://fluidproject.org/blog/2008/01/09/getting-setting-and-removing-tabindex-values-with-javascript/\n\
\t\t\t\t// Use proper attribute retrieval(#12072)\n\
\t\t\t\tvar tabindex = jQuery.find.attr( elem, \"tabindex\" );\n\
\n\
\t\t\t\treturn tabindex ?\n\
\t\t\t\t\tparseInt( tabindex, 10 ) :\n\
\t\t\t\t\trfocusable.test( elem.nodeName ) || rclickable.test( elem.nodeName ) && elem.href ?\n\
\t\t\t\t\t\t0 :\n\
\t\t\t\t\t\t-1;\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
});\n\
\n\
// Some attributes require a special call on IE\n\
// http://msdn.microsoft.com/en-us/library/ms536429%28VS.85%29.aspx\n\
if ( !support.hrefNormalized ) {\n\
\t// href/src property should get the full normalized URL (#10299/#12915)\n\
\tjQuery.each([ \"href\", \"src\" ], function( i, name ) {\n\
\t\tjQuery.propHooks[ name ] = {\n\
\t\t\tget: function( elem ) {\n\
\t\t\t\treturn elem.getAttribute( name, 4 );\n\
\t\t\t}\n\
\t\t};\n\
\t});\n\
}\n\
\n\
// Support: Safari, IE9+\n\
// mis-reports the default selected property of an option\n\
// Accessing the parent's selectedIndex property fixes it\n\
if ( !support.optSelected ) {\n\
\tjQuery.propHooks.selected = {\n\
\t\tget: function( elem ) {\n\
\t\t\tvar parent = elem.parentNode;\n\
\n\
\t\t\tif ( parent ) {\n\
\t\t\t\tparent.selectedIndex;\n\
\n\
\t\t\t\t// Make sure that it also works with optgroups, see #5701\n\
\t\t\t\tif ( parent.parentNode ) {\n\
\t\t\t\t\tparent.parentNode.selectedIndex;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t\treturn null;\n\
\t\t}\n\
\t};\n\
}\n\
\n\
jQuery.each([\n\
\t\"tabIndex\",\n\
\t\"readOnly\",\n\
\t\"maxLength\",\n\
\t\"cellSpacing\",\n\
\t\"cellPadding\",\n\
\t\"rowSpan\",\n\
\t\"colSpan\",\n\
\t\"useMap\",\n\
\t\"frameBorder\",\n\
\t\"contentEditable\"\n\
], function() {\n\
\tjQuery.propFix[ this.toLowerCase() ] = this;\n\
});\n\
\n\
// IE6/7 call enctype encoding\n\
if ( !support.enctype ) {\n\
\tjQuery.propFix.enctype = \"encoding\";\n\
}\n\
\n\
\n\
\n\
\n\
var rclass = /[\\t\\r\\n\
\\f]/g;\n\
\n\
jQuery.fn.extend({\n\
\taddClass: function( value ) {\n\
\t\tvar classes, elem, cur, clazz, j, finalValue,\n\
\t\t\ti = 0,\n\
\t\t\tlen = this.length,\n\
\t\t\tproceed = typeof value === \"string\" && value;\n\
\n\
\t\tif ( jQuery.isFunction( value ) ) {\n\
\t\t\treturn this.each(function( j ) {\n\
\t\t\t\tjQuery( this ).addClass( value.call( this, j, this.className ) );\n\
\t\t\t});\n\
\t\t}\n\
\n\
\t\tif ( proceed ) {\n\
\t\t\t// The disjunction here is for better compressibility (see removeClass)\n\
\t\t\tclasses = ( value || \"\" ).match( rnotwhite ) || [];\n\
\n\
\t\t\tfor ( ; i < len; i++ ) {\n\
\t\t\t\telem = this[ i ];\n\
\t\t\t\tcur = elem.nodeType === 1 && ( elem.className ?\n\
\t\t\t\t\t( \" \" + elem.className + \" \" ).replace( rclass, \" \" ) :\n\
\t\t\t\t\t\" \"\n\
\t\t\t\t);\n\
\n\
\t\t\t\tif ( cur ) {\n\
\t\t\t\t\tj = 0;\n\
\t\t\t\t\twhile ( (clazz = classes[j++]) ) {\n\
\t\t\t\t\t\tif ( cur.indexOf( \" \" + clazz + \" \" ) < 0 ) {\n\
\t\t\t\t\t\t\tcur += clazz + \" \";\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// only assign if different to avoid unneeded rendering.\n\
\t\t\t\t\tfinalValue = jQuery.trim( cur );\n\
\t\t\t\t\tif ( elem.className !== finalValue ) {\n\
\t\t\t\t\t\telem.className = finalValue;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn this;\n\
\t},\n\
\n\
\tremoveClass: function( value ) {\n\
\t\tvar classes, elem, cur, clazz, j, finalValue,\n\
\t\t\ti = 0,\n\
\t\t\tlen = this.length,\n\
\t\t\tproceed = arguments.length === 0 || typeof value === \"string\" && value;\n\
\n\
\t\tif ( jQuery.isFunction( value ) ) {\n\
\t\t\treturn this.each(function( j ) {\n\
\t\t\t\tjQuery( this ).removeClass( value.call( this, j, this.className ) );\n\
\t\t\t});\n\
\t\t}\n\
\t\tif ( proceed ) {\n\
\t\t\tclasses = ( value || \"\" ).match( rnotwhite ) || [];\n\
\n\
\t\t\tfor ( ; i < len; i++ ) {\n\
\t\t\t\telem = this[ i ];\n\
\t\t\t\t// This expression is here for better compressibility (see addClass)\n\
\t\t\t\tcur = elem.nodeType === 1 && ( elem.className ?\n\
\t\t\t\t\t( \" \" + elem.className + \" \" ).replace( rclass, \" \" ) :\n\
\t\t\t\t\t\"\"\n\
\t\t\t\t);\n\
\n\
\t\t\t\tif ( cur ) {\n\
\t\t\t\t\tj = 0;\n\
\t\t\t\t\twhile ( (clazz = classes[j++]) ) {\n\
\t\t\t\t\t\t// Remove *all* instances\n\
\t\t\t\t\t\twhile ( cur.indexOf( \" \" + clazz + \" \" ) >= 0 ) {\n\
\t\t\t\t\t\t\tcur = cur.replace( \" \" + clazz + \" \", \" \" );\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// only assign if different to avoid unneeded rendering.\n\
\t\t\t\t\tfinalValue = value ? jQuery.trim( cur ) : \"\";\n\
\t\t\t\t\tif ( elem.className !== finalValue ) {\n\
\t\t\t\t\t\telem.className = finalValue;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn this;\n\
\t},\n\
\n\
\ttoggleClass: function( value, stateVal ) {\n\
\t\tvar type = typeof value;\n\
\n\
\t\tif ( typeof stateVal === \"boolean\" && type === \"string\" ) {\n\
\t\t\treturn stateVal ? this.addClass( value ) : this.removeClass( value );\n\
\t\t}\n\
\n\
\t\tif ( jQuery.isFunction( value ) ) {\n\
\t\t\treturn this.each(function( i ) {\n\
\t\t\t\tjQuery( this ).toggleClass( value.call(this, i, this.className, stateVal), stateVal );\n\
\t\t\t});\n\
\t\t}\n\
\n\
\t\treturn this.each(function() {\n\
\t\t\tif ( type === \"string\" ) {\n\
\t\t\t\t// toggle individual class names\n\
\t\t\t\tvar className,\n\
\t\t\t\t\ti = 0,\n\
\t\t\t\t\tself = jQuery( this ),\n\
\t\t\t\t\tclassNames = value.match( rnotwhite ) || [];\n\
\n\
\t\t\t\twhile ( (className = classNames[ i++ ]) ) {\n\
\t\t\t\t\t// check each className given, space separated list\n\
\t\t\t\t\tif ( self.hasClass( className ) ) {\n\
\t\t\t\t\t\tself.removeClass( className );\n\
\t\t\t\t\t} else {\n\
\t\t\t\t\t\tself.addClass( className );\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\n\
\t\t\t// Toggle whole class name\n\
\t\t\t} else if ( type === strundefined || type === \"boolean\" ) {\n\
\t\t\t\tif ( this.className ) {\n\
\t\t\t\t\t// store className if set\n\
\t\t\t\t\tjQuery._data( this, \"__className__\", this.className );\n\
\t\t\t\t}\n\
\n\
\t\t\t\t// If the element has a class name or if we're passed \"false\",\n\
\t\t\t\t// then remove the whole classname (if there was one, the above saved it).\n\
\t\t\t\t// Otherwise bring back whatever was previously saved (if anything),\n\
\t\t\t\t// falling back to the empty string if nothing was stored.\n\
\t\t\t\tthis.className = this.className || value === false ? \"\" : jQuery._data( this, \"__className__\" ) || \"\";\n\
\t\t\t}\n\
\t\t});\n\
\t},\n\
\n\
\thasClass: function( selector ) {\n\
\t\tvar className = \" \" + selector + \" \",\n\
\t\t\ti = 0,\n\
\t\t\tl = this.length;\n\
\t\tfor ( ; i < l; i++ ) {\n\
\t\t\tif ( this[i].nodeType === 1 && (\" \" + this[i].className + \" \").replace(rclass, \" \").indexOf( className ) >= 0 ) {\n\
\t\t\t\treturn true;\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn false;\n\
\t}\n\
});\n\
\n\
\n\
\n\
\n\
// Return jQuery for attributes-only inclusion\n\
\n\
\n\
jQuery.each( (\"blur focus focusin focusout load resize scroll unload click dblclick \" +\n\
\t\"mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave \" +\n\
\t\"change select submit keydown keypress keyup error contextmenu\").split(\" \"), function( i, name ) {\n\
\n\
\t// Handle event binding\n\
\tjQuery.fn[ name ] = function( data, fn ) {\n\
\t\treturn arguments.length > 0 ?\n\
\t\t\tthis.on( name, null, data, fn ) :\n\
\t\t\tthis.trigger( name );\n\
\t};\n\
});\n\
\n\
jQuery.fn.extend({\n\
\thover: function( fnOver, fnOut ) {\n\
\t\treturn this.mouseenter( fnOver ).mouseleave( fnOut || fnOver );\n\
\t},\n\
\n\
\tbind: function( types, data, fn ) {\n\
\t\treturn this.on( types, null, data, fn );\n\
\t},\n\
\tunbind: function( types, fn ) {\n\
\t\treturn this.off( types, null, fn );\n\
\t},\n\
\n\
\tdelegate: function( selector, types, data, fn ) {\n\
\t\treturn this.on( types, selector, data, fn );\n\
\t},\n\
\tundelegate: function( selector, types, fn ) {\n\
\t\t// ( namespace ) or ( selector, types [, fn] )\n\
\t\treturn arguments.length === 1 ? this.off( selector, \"**\" ) : this.off( types, selector || \"**\", fn );\n\
\t}\n\
});\n\
\n\
\n\
var nonce = jQuery.now();\n\
\n\
var rquery = (/\\?/);\n\
\n\
\n\
\n\
var rvalidtokens = /(,)|(\\[|{)|(}|])|\"(?:[^\"\\\\\\r\\n\
]|\\\\[\"\\\\\\/bfnrt]|\\\\u[\\da-fA-F]{4})*\"\\s*:?|true|false|null|-?(?!0\\d)\\d+(?:\\.\\d+|)(?:[eE][+-]?\\d+|)/g;\n\
\n\
jQuery.parseJSON = function( data ) {\n\
\t// Attempt to parse using the native JSON parser first\n\
\tif ( window.JSON && window.JSON.parse ) {\n\
\t\t// Support: Android 2.3\n\
\t\t// Workaround failure to string-cast null input\n\
\t\treturn window.JSON.parse( data + \"\" );\n\
\t}\n\
\n\
\tvar requireNonComma,\n\
\t\tdepth = null,\n\
\t\tstr = jQuery.trim( data + \"\" );\n\
\n\
\t// Guard against invalid (and possibly dangerous) input by ensuring that nothing remains\n\
\t// after removing valid tokens\n\
\treturn str && !jQuery.trim( str.replace( rvalidtokens, function( token, comma, open, close ) {\n\
\n\
\t\t// Force termination if we see a misplaced comma\n\
\t\tif ( requireNonComma && comma ) {\n\
\t\t\tdepth = 0;\n\
\t\t}\n\
\n\
\t\t// Perform no more replacements after returning to outermost depth\n\
\t\tif ( depth === 0 ) {\n\
\t\t\treturn token;\n\
\t\t}\n\
\n\
\t\t// Commas must not follow \"[\", \"{\", or \",\"\n\
\t\trequireNonComma = open || comma;\n\
\n\
\t\t// Determine new depth\n\
\t\t// array/object open (\"[\" or \"{\"): depth += true - false (increment)\n\
\t\t// array/object close (\"]\" or \"}\"): depth += false - true (decrement)\n\
\t\t// other cases (\",\" or primitive): depth += true - true (numeric cast)\n\
\t\tdepth += !close - !open;\n\
\n\
\t\t// Remove this token\n\
\t\treturn \"\";\n\
\t}) ) ?\n\
\t\t( Function( \"return \" + str ) )() :\n\
\t\tjQuery.error( \"Invalid JSON: \" + data );\n\
};\n\
\n\
\n\
// Cross-browser xml parsing\n\
jQuery.parseXML = function( data ) {\n\
\tvar xml, tmp;\n\
\tif ( !data || typeof data !== \"string\" ) {\n\
\t\treturn null;\n\
\t}\n\
\ttry {\n\
\t\tif ( window.DOMParser ) { // Standard\n\
\t\t\ttmp = new DOMParser();\n\
\t\t\txml = tmp.parseFromString( data, \"text/xml\" );\n\
\t\t} else { // IE\n\
\t\t\txml = new ActiveXObject( \"Microsoft.XMLDOM\" );\n\
\t\t\txml.async = \"false\";\n\
\t\t\txml.loadXML( data );\n\
\t\t}\n\
\t} catch( e ) {\n\
\t\txml = undefined;\n\
\t}\n\
\tif ( !xml || !xml.documentElement || xml.getElementsByTagName( \"parsererror\" ).length ) {\n\
\t\tjQuery.error( \"Invalid XML: \" + data );\n\
\t}\n\
\treturn xml;\n\
};\n\
\n\
\n\
var\n\
\t// Document location\n\
\tajaxLocParts,\n\
\tajaxLocation,\n\
\n\
\trhash = /#.*$/,\n\
\trts = /([?&])_=[^&]*/,\n\
\trheaders = /^(.*?):[ \\t]*([^\\r\\n\
]*)\\r?$/mg, // IE leaves an \\r character at EOL\n\
\t// #7653, #8125, #8152: local protocol detection\n\
\trlocalProtocol = /^(?:about|app|app-storage|.+-extension|file|res|widget):$/,\n\
\trnoContent = /^(?:GET|HEAD)$/,\n\
\trprotocol = /^\\/\\//,\n\
\trurl = /^([\\w.+-]+:)(?:\\/\\/(?:[^\\/?#]*@|)([^\\/?#:]*)(?::(\\d+)|)|)/,\n\
\n\
\t/* Prefilters\n\
\t * 1) They are useful to introduce custom dataTypes (see ajax/jsonp.js for an example)\n\
\t * 2) These are called:\n\
\t *    - BEFORE asking for a transport\n\
\t *    - AFTER param serialization (s.data is a string if s.processData is true)\n\
\t * 3) key is the dataType\n\
\t * 4) the catchall symbol \"*\" can be used\n\
\t * 5) execution will start with transport dataType and THEN continue down to \"*\" if needed\n\
\t */\n\
\tprefilters = {},\n\
\n\
\t/* Transports bindings\n\
\t * 1) key is the dataType\n\
\t * 2) the catchall symbol \"*\" can be used\n\
\t * 3) selection will start with transport dataType and THEN go to \"*\" if needed\n\
\t */\n\
\ttransports = {},\n\
\n\
\t// Avoid comment-prolog char sequence (#10098); must appease lint and evade compression\n\
\tallTypes = \"*/\".concat(\"*\");\n\
\n\
// #8138, IE may throw an exception when accessing\n\
// a field from window.location if document.domain has been set\n\
try {\n\
\tajaxLocation = location.href;\n\
} catch( e ) {\n\
\t// Use the href attribute of an A element\n\
\t// since IE will modify it given document.location\n\
\tajaxLocation = document.createElement( \"a\" );\n\
\tajaxLocation.href = \"\";\n\
\tajaxLocation = ajaxLocation.href;\n\
}\n\
\n\
// Segment location into parts\n\
ajaxLocParts = rurl.exec( ajaxLocation.toLowerCase() ) || [];\n\
\n\
// Base \"constructor\" for jQuery.ajaxPrefilter and jQuery.ajaxTransport\n\
function addToPrefiltersOrTransports( structure ) {\n\
\n\
\t// dataTypeExpression is optional and defaults to \"*\"\n\
\treturn function( dataTypeExpression, func ) {\n\
\n\
\t\tif ( typeof dataTypeExpression !== \"string\" ) {\n\
\t\t\tfunc = dataTypeExpression;\n\
\t\t\tdataTypeExpression = \"*\";\n\
\t\t}\n\
\n\
\t\tvar dataType,\n\
\t\t\ti = 0,\n\
\t\t\tdataTypes = dataTypeExpression.toLowerCase().match( rnotwhite ) || [];\n\
\n\
\t\tif ( jQuery.isFunction( func ) ) {\n\
\t\t\t// For each dataType in the dataTypeExpression\n\
\t\t\twhile ( (dataType = dataTypes[i++]) ) {\n\
\t\t\t\t// Prepend if requested\n\
\t\t\t\tif ( dataType.charAt( 0 ) === \"+\" ) {\n\
\t\t\t\t\tdataType = dataType.slice( 1 ) || \"*\";\n\
\t\t\t\t\t(structure[ dataType ] = structure[ dataType ] || []).unshift( func );\n\
\n\
\t\t\t\t// Otherwise append\n\
\t\t\t\t} else {\n\
\t\t\t\t\t(structure[ dataType ] = structure[ dataType ] || []).push( func );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\t};\n\
}\n\
\n\
// Base inspection function for prefilters and transports\n\
function inspectPrefiltersOrTransports( structure, options, originalOptions, jqXHR ) {\n\
\n\
\tvar inspected = {},\n\
\t\tseekingTransport = ( structure === transports );\n\
\n\
\tfunction inspect( dataType ) {\n\
\t\tvar selected;\n\
\t\tinspected[ dataType ] = true;\n\
\t\tjQuery.each( structure[ dataType ] || [], function( _, prefilterOrFactory ) {\n\
\t\t\tvar dataTypeOrTransport = prefilterOrFactory( options, originalOptions, jqXHR );\n\
\t\t\tif ( typeof dataTypeOrTransport === \"string\" && !seekingTransport && !inspected[ dataTypeOrTransport ] ) {\n\
\t\t\t\toptions.dataTypes.unshift( dataTypeOrTransport );\n\
\t\t\t\tinspect( dataTypeOrTransport );\n\
\t\t\t\treturn false;\n\
\t\t\t} else if ( seekingTransport ) {\n\
\t\t\t\treturn !( selected = dataTypeOrTransport );\n\
\t\t\t}\n\
\t\t});\n\
\t\treturn selected;\n\
\t}\n\
\n\
\treturn inspect( options.dataTypes[ 0 ] ) || !inspected[ \"*\" ] && inspect( \"*\" );\n\
}\n\
\n\
// A special extend for ajax options\n\
// that takes \"flat\" options (not to be deep extended)\n\
// Fixes #9887\n\
function ajaxExtend( target, src ) {\n\
\tvar deep, key,\n\
\t\tflatOptions = jQuery.ajaxSettings.flatOptions || {};\n\
\n\
\tfor ( key in src ) {\n\
\t\tif ( src[ key ] !== undefined ) {\n\
\t\t\t( flatOptions[ key ] ? target : ( deep || (deep = {}) ) )[ key ] = src[ key ];\n\
\t\t}\n\
\t}\n\
\tif ( deep ) {\n\
\t\tjQuery.extend( true, target, deep );\n\
\t}\n\
\n\
\treturn target;\n\
}\n\
\n\
/* Handles responses to an ajax request:\n\
 * - finds the right dataType (mediates between content-type and expected dataType)\n\
 * - returns the corresponding response\n\
 */\n\
function ajaxHandleResponses( s, jqXHR, responses ) {\n\
\tvar firstDataType, ct, finalDataType, type,\n\
\t\tcontents = s.contents,\n\
\t\tdataTypes = s.dataTypes;\n\
\n\
\t// Remove auto dataType and get content-type in the process\n\
\twhile ( dataTypes[ 0 ] === \"*\" ) {\n\
\t\tdataTypes.shift();\n\
\t\tif ( ct === undefined ) {\n\
\t\t\tct = s.mimeType || jqXHR.getResponseHeader(\"Content-Type\");\n\
\t\t}\n\
\t}\n\
\n\
\t// Check if we're dealing with a known content-type\n\
\tif ( ct ) {\n\
\t\tfor ( type in contents ) {\n\
\t\t\tif ( contents[ type ] && contents[ type ].test( ct ) ) {\n\
\t\t\t\tdataTypes.unshift( type );\n\
\t\t\t\tbreak;\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\t// Check to see if we have a response for the expected dataType\n\
\tif ( dataTypes[ 0 ] in responses ) {\n\
\t\tfinalDataType = dataTypes[ 0 ];\n\
\t} else {\n\
\t\t// Try convertible dataTypes\n\
\t\tfor ( type in responses ) {\n\
\t\t\tif ( !dataTypes[ 0 ] || s.converters[ type + \" \" + dataTypes[0] ] ) {\n\
\t\t\t\tfinalDataType = type;\n\
\t\t\t\tbreak;\n\
\t\t\t}\n\
\t\t\tif ( !firstDataType ) {\n\
\t\t\t\tfirstDataType = type;\n\
\t\t\t}\n\
\t\t}\n\
\t\t// Or just use first one\n\
\t\tfinalDataType = finalDataType || firstDataType;\n\
\t}\n\
\n\
\t// If we found a dataType\n\
\t// We add the dataType to the list if needed\n\
\t// and return the corresponding response\n\
\tif ( finalDataType ) {\n\
\t\tif ( finalDataType !== dataTypes[ 0 ] ) {\n\
\t\t\tdataTypes.unshift( finalDataType );\n\
\t\t}\n\
\t\treturn responses[ finalDataType ];\n\
\t}\n\
}\n\
\n\
/* Chain conversions given the request and the original response\n\
 * Also sets the responseXXX fields on the jqXHR instance\n\
 */\n\
function ajaxConvert( s, response, jqXHR, isSuccess ) {\n\
\tvar conv2, current, conv, tmp, prev,\n\
\t\tconverters = {},\n\
\t\t// Work with a copy of dataTypes in case we need to modify it for conversion\n\
\t\tdataTypes = s.dataTypes.slice();\n\
\n\
\t// Create converters map with lowercased keys\n\
\tif ( dataTypes[ 1 ] ) {\n\
\t\tfor ( conv in s.converters ) {\n\
\t\t\tconverters[ conv.toLowerCase() ] = s.converters[ conv ];\n\
\t\t}\n\
\t}\n\
\n\
\tcurrent = dataTypes.shift();\n\
\n\
\t// Convert to each sequential dataType\n\
\twhile ( current ) {\n\
\n\
\t\tif ( s.responseFields[ current ] ) {\n\
\t\t\tjqXHR[ s.responseFields[ current ] ] = response;\n\
\t\t}\n\
\n\
\t\t// Apply the dataFilter if provided\n\
\t\tif ( !prev && isSuccess && s.dataFilter ) {\n\
\t\t\tresponse = s.dataFilter( response, s.dataType );\n\
\t\t}\n\
\n\
\t\tprev = current;\n\
\t\tcurrent = dataTypes.shift();\n\
\n\
\t\tif ( current ) {\n\
\n\
\t\t\t// There's only work to do if current dataType is non-auto\n\
\t\t\tif ( current === \"*\" ) {\n\
\n\
\t\t\t\tcurrent = prev;\n\
\n\
\t\t\t// Convert response if prev dataType is non-auto and differs from current\n\
\t\t\t} else if ( prev !== \"*\" && prev !== current ) {\n\
\n\
\t\t\t\t// Seek a direct converter\n\
\t\t\t\tconv = converters[ prev + \" \" + current ] || converters[ \"* \" + current ];\n\
\n\
\t\t\t\t// If none found, seek a pair\n\
\t\t\t\tif ( !conv ) {\n\
\t\t\t\t\tfor ( conv2 in converters ) {\n\
\n\
\t\t\t\t\t\t// If conv2 outputs current\n\
\t\t\t\t\t\ttmp = conv2.split( \" \" );\n\
\t\t\t\t\t\tif ( tmp[ 1 ] === current ) {\n\
\n\
\t\t\t\t\t\t\t// If prev can be converted to accepted input\n\
\t\t\t\t\t\t\tconv = converters[ prev + \" \" + tmp[ 0 ] ] ||\n\
\t\t\t\t\t\t\t\tconverters[ \"* \" + tmp[ 0 ] ];\n\
\t\t\t\t\t\t\tif ( conv ) {\n\
\t\t\t\t\t\t\t\t// Condense equivalence converters\n\
\t\t\t\t\t\t\t\tif ( conv === true ) {\n\
\t\t\t\t\t\t\t\t\tconv = converters[ conv2 ];\n\
\n\
\t\t\t\t\t\t\t\t// Otherwise, insert the intermediate dataType\n\
\t\t\t\t\t\t\t\t} else if ( converters[ conv2 ] !== true ) {\n\
\t\t\t\t\t\t\t\t\tcurrent = tmp[ 0 ];\n\
\t\t\t\t\t\t\t\t\tdataTypes.unshift( tmp[ 1 ] );\n\
\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t\tbreak;\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\n\
\t\t\t\t// Apply converter (if not an equivalence)\n\
\t\t\t\tif ( conv !== true ) {\n\
\n\
\t\t\t\t\t// Unless errors are allowed to bubble, catch and return them\n\
\t\t\t\t\tif ( conv && s[ \"throws\" ] ) {\n\
\t\t\t\t\t\tresponse = conv( response );\n\
\t\t\t\t\t} else {\n\
\t\t\t\t\t\ttry {\n\
\t\t\t\t\t\t\tresponse = conv( response );\n\
\t\t\t\t\t\t} catch ( e ) {\n\
\t\t\t\t\t\t\treturn { state: \"parsererror\", error: conv ? e : \"No conversion from \" + prev + \" to \" + current };\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\t}\n\
\n\
\treturn { state: \"success\", data: response };\n\
}\n\
\n\
jQuery.extend({\n\
\n\
\t// Counter for holding the number of active queries\n\
\tactive: 0,\n\
\n\
\t// Last-Modified header cache for next request\n\
\tlastModified: {},\n\
\tetag: {},\n\
\n\
\tajaxSettings: {\n\
\t\turl: ajaxLocation,\n\
\t\ttype: \"GET\",\n\
\t\tisLocal: rlocalProtocol.test( ajaxLocParts[ 1 ] ),\n\
\t\tglobal: true,\n\
\t\tprocessData: true,\n\
\t\tasync: true,\n\
\t\tcontentType: \"application/x-www-form-urlencoded; charset=UTF-8\",\n\
\t\t/*\n\
\t\ttimeout: 0,\n\
\t\tdata: null,\n\
\t\tdataType: null,\n\
\t\tusername: null,\n\
\t\tpassword: null,\n\
\t\tcache: null,\n\
\t\tthrows: false,\n\
\t\ttraditional: false,\n\
\t\theaders: {},\n\
\t\t*/\n\
\n\
\t\taccepts: {\n\
\t\t\t\"*\": allTypes,\n\
\t\t\ttext: \"text/plain\",\n\
\t\t\thtml: \"text/html\",\n\
\t\t\txml: \"application/xml, text/xml\",\n\
\t\t\tjson: \"application/json, text/javascript\"\n\
\t\t},\n\
\n\
\t\tcontents: {\n\
\t\t\txml: /xml/,\n\
\t\t\thtml: /html/,\n\
\t\t\tjson: /json/\n\
\t\t},\n\
\n\
\t\tresponseFields: {\n\
\t\t\txml: \"responseXML\",\n\
\t\t\ttext: \"responseText\",\n\
\t\t\tjson: \"responseJSON\"\n\
\t\t},\n\
\n\
\t\t// Data converters\n\
\t\t// Keys separate source (or catchall \"*\") and destination types with a single space\n\
\t\tconverters: {\n\
\n\
\t\t\t// Convert anything to text\n\
\t\t\t\"* text\": String,\n\
\n\
\t\t\t// Text to html (true = no transformation)\n\
\t\t\t\"text html\": true,\n\
\n\
\t\t\t// Evaluate text as a json expression\n\
\t\t\t\"text json\": jQuery.parseJSON,\n\
\n\
\t\t\t// Parse text as xml\n\
\t\t\t\"text xml\": jQuery.parseXML\n\
\t\t},\n\
\n\
\t\t// For options that shouldn't be deep extended:\n\
\t\t// you can add your own custom options here if\n\
\t\t// and when you create one that shouldn't be\n\
\t\t// deep extended (see ajaxExtend)\n\
\t\tflatOptions: {\n\
\t\t\turl: true,\n\
\t\t\tcontext: true\n\
\t\t}\n\
\t},\n\
\n\
\t// Creates a full fledged settings object into target\n\
\t// with both ajaxSettings and settings fields.\n\
\t// If target is omitted, writes into ajaxSettings.\n\
\tajaxSetup: function( target, settings ) {\n\
\t\treturn settings ?\n\
\n\
\t\t\t// Building a settings object\n\
\t\t\tajaxExtend( ajaxExtend( target, jQuery.ajaxSettings ), settings ) :\n\
\n\
\t\t\t// Extending ajaxSettings\n\
\t\t\tajaxExtend( jQuery.ajaxSettings, target );\n\
\t},\n\
\n\
\tajaxPrefilter: addToPrefiltersOrTransports( prefilters ),\n\
\tajaxTransport: addToPrefiltersOrTransports( transports ),\n\
\n\
\t// Main method\n\
\tajax: function( url, options ) {\n\
\n\
\t\t// If url is an object, simulate pre-1.5 signature\n\
\t\tif ( typeof url === \"object\" ) {\n\
\t\t\toptions = url;\n\
\t\t\turl = undefined;\n\
\t\t}\n\
\n\
\t\t// Force options to be an object\n\
\t\toptions = options || {};\n\
\n\
\t\tvar // Cross-domain detection vars\n\
\t\t\tparts,\n\
\t\t\t// Loop variable\n\
\t\t\ti,\n\
\t\t\t// URL without anti-cache param\n\
\t\t\tcacheURL,\n\
\t\t\t// Response headers as string\n\
\t\t\tresponseHeadersString,\n\
\t\t\t// timeout handle\n\
\t\t\ttimeoutTimer,\n\
\n\
\t\t\t// To know if global events are to be dispatched\n\
\t\t\tfireGlobals,\n\
\n\
\t\t\ttransport,\n\
\t\t\t// Response headers\n\
\t\t\tresponseHeaders,\n\
\t\t\t// Create the final options object\n\
\t\t\ts = jQuery.ajaxSetup( {}, options ),\n\
\t\t\t// Callbacks context\n\
\t\t\tcallbackContext = s.context || s,\n\
\t\t\t// Context for global events is callbackContext if it is a DOM node or jQuery collection\n\
\t\t\tglobalEventContext = s.context && ( callbackContext.nodeType || callbackContext.jquery ) ?\n\
\t\t\t\tjQuery( callbackContext ) :\n\
\t\t\t\tjQuery.event,\n\
\t\t\t// Deferreds\n\
\t\t\tdeferred = jQuery.Deferred(),\n\
\t\t\tcompleteDeferred = jQuery.Callbacks(\"once memory\"),\n\
\t\t\t// Status-dependent callbacks\n\
\t\t\tstatusCode = s.statusCode || {},\n\
\t\t\t// Headers (they are sent all at once)\n\
\t\t\trequestHeaders = {},\n\
\t\t\trequestHeadersNames = {},\n\
\t\t\t// The jqXHR state\n\
\t\t\tstate = 0,\n\
\t\t\t// Default abort message\n\
\t\t\tstrAbort = \"canceled\",\n\
\t\t\t// Fake xhr\n\
\t\t\tjqXHR = {\n\
\t\t\t\treadyState: 0,\n\
\n\
\t\t\t\t// Builds headers hashtable if needed\n\
\t\t\t\tgetResponseHeader: function( key ) {\n\
\t\t\t\t\tvar match;\n\
\t\t\t\t\tif ( state === 2 ) {\n\
\t\t\t\t\t\tif ( !responseHeaders ) {\n\
\t\t\t\t\t\t\tresponseHeaders = {};\n\
\t\t\t\t\t\t\twhile ( (match = rheaders.exec( responseHeadersString )) ) {\n\
\t\t\t\t\t\t\t\tresponseHeaders[ match[1].toLowerCase() ] = match[ 2 ];\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t\tmatch = responseHeaders[ key.toLowerCase() ];\n\
\t\t\t\t\t}\n\
\t\t\t\t\treturn match == null ? null : match;\n\
\t\t\t\t},\n\
\n\
\t\t\t\t// Raw string\n\
\t\t\t\tgetAllResponseHeaders: function() {\n\
\t\t\t\t\treturn state === 2 ? responseHeadersString : null;\n\
\t\t\t\t},\n\
\n\
\t\t\t\t// Caches the header\n\
\t\t\t\tsetRequestHeader: function( name, value ) {\n\
\t\t\t\t\tvar lname = name.toLowerCase();\n\
\t\t\t\t\tif ( !state ) {\n\
\t\t\t\t\t\tname = requestHeadersNames[ lname ] = requestHeadersNames[ lname ] || name;\n\
\t\t\t\t\t\trequestHeaders[ name ] = value;\n\
\t\t\t\t\t}\n\
\t\t\t\t\treturn this;\n\
\t\t\t\t},\n\
\n\
\t\t\t\t// Overrides response content-type header\n\
\t\t\t\toverrideMimeType: function( type ) {\n\
\t\t\t\t\tif ( !state ) {\n\
\t\t\t\t\t\ts.mimeType = type;\n\
\t\t\t\t\t}\n\
\t\t\t\t\treturn this;\n\
\t\t\t\t},\n\
\n\
\t\t\t\t// Status-dependent callbacks\n\
\t\t\t\tstatusCode: function( map ) {\n\
\t\t\t\t\tvar code;\n\
\t\t\t\t\tif ( map ) {\n\
\t\t\t\t\t\tif ( state < 2 ) {\n\
\t\t\t\t\t\t\tfor ( code in map ) {\n\
\t\t\t\t\t\t\t\t// Lazy-add the new callback in a way that preserves old ones\n\
\t\t\t\t\t\t\t\tstatusCode[ code ] = [ statusCode[ code ], map[ code ] ];\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t} else {\n\
\t\t\t\t\t\t\t// Execute the appropriate callbacks\n\
\t\t\t\t\t\t\tjqXHR.always( map[ jqXHR.status ] );\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t\treturn this;\n\
\t\t\t\t},\n\
\n\
\t\t\t\t// Cancel the request\n\
\t\t\t\tabort: function( statusText ) {\n\
\t\t\t\t\tvar finalText = statusText || strAbort;\n\
\t\t\t\t\tif ( transport ) {\n\
\t\t\t\t\t\ttransport.abort( finalText );\n\
\t\t\t\t\t}\n\
\t\t\t\t\tdone( 0, finalText );\n\
\t\t\t\t\treturn this;\n\
\t\t\t\t}\n\
\t\t\t};\n\
\n\
\t\t// Attach deferreds\n\
\t\tdeferred.promise( jqXHR ).complete = completeDeferred.add;\n\
\t\tjqXHR.success = jqXHR.done;\n\
\t\tjqXHR.error = jqXHR.fail;\n\
\n\
\t\t// Remove hash character (#7531: and string promotion)\n\
\t\t// Add protocol if not provided (#5866: IE7 issue with protocol-less urls)\n\
\t\t// Handle falsy url in the settings object (#10093: consistency with old signature)\n\
\t\t// We also use the url parameter if available\n\
\t\ts.url = ( ( url || s.url || ajaxLocation ) + \"\" ).replace( rhash, \"\" ).replace( rprotocol, ajaxLocParts[ 1 ] + \"//\" );\n\
\n\
\t\t// Alias method option to type as per ticket #12004\n\
\t\ts.type = options.method || options.type || s.method || s.type;\n\
\n\
\t\t// Extract dataTypes list\n\
\t\ts.dataTypes = jQuery.trim( s.dataType || \"*\" ).toLowerCase().match( rnotwhite ) || [ \"\" ];\n\
\n\
\t\t// A cross-domain request is in order when we have a protocol:host:port mismatch\n\
\t\tif ( s.crossDomain == null ) {\n\
\t\t\tparts = rurl.exec( s.url.toLowerCase() );\n\
\t\t\ts.crossDomain = !!( parts &&\n\
\t\t\t\t( parts[ 1 ] !== ajaxLocParts[ 1 ] || parts[ 2 ] !== ajaxLocParts[ 2 ] ||\n\
\t\t\t\t\t( parts[ 3 ] || ( parts[ 1 ] === \"http:\" ? \"80\" : \"443\" ) ) !==\n\
\t\t\t\t\t\t( ajaxLocParts[ 3 ] || ( ajaxLocParts[ 1 ] === \"http:\" ? \"80\" : \"443\" ) ) )\n\
\t\t\t);\n\
\t\t}\n\
\n\
\t\t// Convert data if not already a string\n\
\t\tif ( s.data && s.processData && typeof s.data !== \"string\" ) {\n\
\t\t\ts.data = jQuery.param( s.data, s.traditional );\n\
\t\t}\n\
\n\
\t\t// Apply prefilters\n\
\t\tinspectPrefiltersOrTransports( prefilters, s, options, jqXHR );\n\
\n\
\t\t// If request was aborted inside a prefilter, stop there\n\
\t\tif ( state === 2 ) {\n\
\t\t\treturn jqXHR;\n\
\t\t}\n\
\n\
\t\t// We can fire global events as of now if asked to\n\
\t\tfireGlobals = s.global;\n\
\n\
\t\t// Watch for a new set of requests\n\
\t\tif ( fireGlobals && jQuery.active++ === 0 ) {\n\
\t\t\tjQuery.event.trigger(\"ajaxStart\");\n\
\t\t}\n\
\n\
\t\t// Uppercase the type\n\
\t\ts.type = s.type.toUpperCase();\n\
\n\
\t\t// Determine if request has content\n\
\t\ts.hasContent = !rnoContent.test( s.type );\n\
\n\
\t\t// Save the URL in case we're toying with the If-Modified-Since\n\
\t\t// and/or If-None-Match header later on\n\
\t\tcacheURL = s.url;\n\
\n\
\t\t// More options handling for requests with no content\n\
\t\tif ( !s.hasContent ) {\n\
\n\
\t\t\t// If data is available, append data to url\n\
\t\t\tif ( s.data ) {\n\
\t\t\t\tcacheURL = ( s.url += ( rquery.test( cacheURL ) ? \"&\" : \"?\" ) + s.data );\n\
\t\t\t\t// #9682: remove data so that it's not used in an eventual retry\n\
\t\t\t\tdelete s.data;\n\
\t\t\t}\n\
\n\
\t\t\t// Add anti-cache in url if needed\n\
\t\t\tif ( s.cache === false ) {\n\
\t\t\t\ts.url = rts.test( cacheURL ) ?\n\
\n\
\t\t\t\t\t// If there is already a '_' parameter, set its value\n\
\t\t\t\t\tcacheURL.replace( rts, \"$1_=\" + nonce++ ) :\n\
\n\
\t\t\t\t\t// Otherwise add one to the end\n\
\t\t\t\t\tcacheURL + ( rquery.test( cacheURL ) ? \"&\" : \"?\" ) + \"_=\" + nonce++;\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Set the If-Modified-Since and/or If-None-Match header, if in ifModified mode.\n\
\t\tif ( s.ifModified ) {\n\
\t\t\tif ( jQuery.lastModified[ cacheURL ] ) {\n\
\t\t\t\tjqXHR.setRequestHeader( \"If-Modified-Since\", jQuery.lastModified[ cacheURL ] );\n\
\t\t\t}\n\
\t\t\tif ( jQuery.etag[ cacheURL ] ) {\n\
\t\t\t\tjqXHR.setRequestHeader( \"If-None-Match\", jQuery.etag[ cacheURL ] );\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Set the correct header, if data is being sent\n\
\t\tif ( s.data && s.hasContent && s.contentType !== false || options.contentType ) {\n\
\t\t\tjqXHR.setRequestHeader( \"Content-Type\", s.contentType );\n\
\t\t}\n\
\n\
\t\t// Set the Accepts header for the server, depending on the dataType\n\
\t\tjqXHR.setRequestHeader(\n\
\t\t\t\"Accept\",\n\
\t\t\ts.dataTypes[ 0 ] && s.accepts[ s.dataTypes[0] ] ?\n\
\t\t\t\ts.accepts[ s.dataTypes[0] ] + ( s.dataTypes[ 0 ] !== \"*\" ? \", \" + allTypes + \"; q=0.01\" : \"\" ) :\n\
\t\t\t\ts.accepts[ \"*\" ]\n\
\t\t);\n\
\n\
\t\t// Check for headers option\n\
\t\tfor ( i in s.headers ) {\n\
\t\t\tjqXHR.setRequestHeader( i, s.headers[ i ] );\n\
\t\t}\n\
\n\
\t\t// Allow custom headers/mimetypes and early abort\n\
\t\tif ( s.beforeSend && ( s.beforeSend.call( callbackContext, jqXHR, s ) === false || state === 2 ) ) {\n\
\t\t\t// Abort if not done already and return\n\
\t\t\treturn jqXHR.abort();\n\
\t\t}\n\
\n\
\t\t// aborting is no longer a cancellation\n\
\t\tstrAbort = \"abort\";\n\
\n\
\t\t// Install callbacks on deferreds\n\
\t\tfor ( i in { success: 1, error: 1, complete: 1 } ) {\n\
\t\t\tjqXHR[ i ]( s[ i ] );\n\
\t\t}\n\
\n\
\t\t// Get transport\n\
\t\ttransport = inspectPrefiltersOrTransports( transports, s, options, jqXHR );\n\
\n\
\t\t// If no transport, we auto-abort\n\
\t\tif ( !transport ) {\n\
\t\t\tdone( -1, \"No Transport\" );\n\
\t\t} else {\n\
\t\t\tjqXHR.readyState = 1;\n\
\n\
\t\t\t// Send global event\n\
\t\t\tif ( fireGlobals ) {\n\
\t\t\t\tglobalEventContext.trigger( \"ajaxSend\", [ jqXHR, s ] );\n\
\t\t\t}\n\
\t\t\t// Timeout\n\
\t\t\tif ( s.async && s.timeout > 0 ) {\n\
\t\t\t\ttimeoutTimer = setTimeout(function() {\n\
\t\t\t\t\tjqXHR.abort(\"timeout\");\n\
\t\t\t\t}, s.timeout );\n\
\t\t\t}\n\
\n\
\t\t\ttry {\n\
\t\t\t\tstate = 1;\n\
\t\t\t\ttransport.send( requestHeaders, done );\n\
\t\t\t} catch ( e ) {\n\
\t\t\t\t// Propagate exception as error if not done\n\
\t\t\t\tif ( state < 2 ) {\n\
\t\t\t\t\tdone( -1, e );\n\
\t\t\t\t// Simply rethrow otherwise\n\
\t\t\t\t} else {\n\
\t\t\t\t\tthrow e;\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\t// Callback for when everything is done\n\
\t\tfunction done( status, nativeStatusText, responses, headers ) {\n\
\t\t\tvar isSuccess, success, error, response, modified,\n\
\t\t\t\tstatusText = nativeStatusText;\n\
\n\
\t\t\t// Called once\n\
\t\t\tif ( state === 2 ) {\n\
\t\t\t\treturn;\n\
\t\t\t}\n\
\n\
\t\t\t// State is \"done\" now\n\
\t\t\tstate = 2;\n\
\n\
\t\t\t// Clear timeout if it exists\n\
\t\t\tif ( timeoutTimer ) {\n\
\t\t\t\tclearTimeout( timeoutTimer );\n\
\t\t\t}\n\
\n\
\t\t\t// Dereference transport for early garbage collection\n\
\t\t\t// (no matter how long the jqXHR object will be used)\n\
\t\t\ttransport = undefined;\n\
\n\
\t\t\t// Cache response headers\n\
\t\t\tresponseHeadersString = headers || \"\";\n\
\n\
\t\t\t// Set readyState\n\
\t\t\tjqXHR.readyState = status > 0 ? 4 : 0;\n\
\n\
\t\t\t// Determine if successful\n\
\t\t\tisSuccess = status >= 200 && status < 300 || status === 304;\n\
\n\
\t\t\t// Get response data\n\
\t\t\tif ( responses ) {\n\
\t\t\t\tresponse = ajaxHandleResponses( s, jqXHR, responses );\n\
\t\t\t}\n\
\n\
\t\t\t// Convert no matter what (that way responseXXX fields are always set)\n\
\t\t\tresponse = ajaxConvert( s, response, jqXHR, isSuccess );\n\
\n\
\t\t\t// If successful, handle type chaining\n\
\t\t\tif ( isSuccess ) {\n\
\n\
\t\t\t\t// Set the If-Modified-Since and/or If-None-Match header, if in ifModified mode.\n\
\t\t\t\tif ( s.ifModified ) {\n\
\t\t\t\t\tmodified = jqXHR.getResponseHeader(\"Last-Modified\");\n\
\t\t\t\t\tif ( modified ) {\n\
\t\t\t\t\t\tjQuery.lastModified[ cacheURL ] = modified;\n\
\t\t\t\t\t}\n\
\t\t\t\t\tmodified = jqXHR.getResponseHeader(\"etag\");\n\
\t\t\t\t\tif ( modified ) {\n\
\t\t\t\t\t\tjQuery.etag[ cacheURL ] = modified;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\n\
\t\t\t\t// if no content\n\
\t\t\t\tif ( status === 204 || s.type === \"HEAD\" ) {\n\
\t\t\t\t\tstatusText = \"nocontent\";\n\
\n\
\t\t\t\t// if not modified\n\
\t\t\t\t} else if ( status === 304 ) {\n\
\t\t\t\t\tstatusText = \"notmodified\";\n\
\n\
\t\t\t\t// If we have data, let's convert it\n\
\t\t\t\t} else {\n\
\t\t\t\t\tstatusText = response.state;\n\
\t\t\t\t\tsuccess = response.data;\n\
\t\t\t\t\terror = response.error;\n\
\t\t\t\t\tisSuccess = !error;\n\
\t\t\t\t}\n\
\t\t\t} else {\n\
\t\t\t\t// We extract error from statusText\n\
\t\t\t\t// then normalize statusText and status for non-aborts\n\
\t\t\t\terror = statusText;\n\
\t\t\t\tif ( status || !statusText ) {\n\
\t\t\t\t\tstatusText = \"error\";\n\
\t\t\t\t\tif ( status < 0 ) {\n\
\t\t\t\t\t\tstatus = 0;\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t}\n\
\n\
\t\t\t// Set data for the fake xhr object\n\
\t\t\tjqXHR.status = status;\n\
\t\t\tjqXHR.statusText = ( nativeStatusText || statusText ) + \"\";\n\
\n\
\t\t\t// Success/Error\n\
\t\t\tif ( isSuccess ) {\n\
\t\t\t\tdeferred.resolveWith( callbackContext, [ success, statusText, jqXHR ] );\n\
\t\t\t} else {\n\
\t\t\t\tdeferred.rejectWith( callbackContext, [ jqXHR, statusText, error ] );\n\
\t\t\t}\n\
\n\
\t\t\t// Status-dependent callbacks\n\
\t\t\tjqXHR.statusCode( statusCode );\n\
\t\t\tstatusCode = undefined;\n\
\n\
\t\t\tif ( fireGlobals ) {\n\
\t\t\t\tglobalEventContext.trigger( isSuccess ? \"ajaxSuccess\" : \"ajaxError\",\n\
\t\t\t\t\t[ jqXHR, s, isSuccess ? success : error ] );\n\
\t\t\t}\n\
\n\
\t\t\t// Complete\n\
\t\t\tcompleteDeferred.fireWith( callbackContext, [ jqXHR, statusText ] );\n\
\n\
\t\t\tif ( fireGlobals ) {\n\
\t\t\t\tglobalEventContext.trigger( \"ajaxComplete\", [ jqXHR, s ] );\n\
\t\t\t\t// Handle the global AJAX counter\n\
\t\t\t\tif ( !( --jQuery.active ) ) {\n\
\t\t\t\t\tjQuery.event.trigger(\"ajaxStop\");\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t}\n\
\n\
\t\treturn jqXHR;\n\
\t},\n\
\n\
\tgetJSON: function( url, data, callback ) {\n\
\t\treturn jQuery.get( url, data, callback, \"json\" );\n\
\t},\n\
\n\
\tgetScript: function( url, callback ) {\n\
\t\treturn jQuery.get( url, undefined, callback, \"script\" );\n\
\t}\n\
});\n\
\n\
jQuery.each( [ \"get\", \"post\" ], function( i, method ) {\n\
\tjQuery[ method ] = function( url, data, callback, type ) {\n\
\t\t// shift arguments if data argument was omitted\n\
\t\tif ( jQuery.isFunction( data ) ) {\n\
\t\t\ttype = type || callback;\n\
\t\t\tcallback = data;\n\
\t\t\tdata = undefined;\n\
\t\t}\n\
\n\
\t\treturn jQuery.ajax({\n\
\t\t\turl: url,\n\
\t\t\ttype: method,\n\
\t\t\tdataType: type,\n\
\t\t\tdata: data,\n\
\t\t\tsuccess: callback\n\
\t\t});\n\
\t};\n\
});\n\
\n\
// Attach a bunch of functions for handling common AJAX events\n\
jQuery.each( [ \"ajaxStart\", \"ajaxStop\", \"ajaxComplete\", \"ajaxError\", \"ajaxSuccess\", \"ajaxSend\" ], function( i, type ) {\n\
\tjQuery.fn[ type ] = function( fn ) {\n\
\t\treturn this.on( type, fn );\n\
\t};\n\
});\n\
\n\
\n\
jQuery._evalUrl = function( url ) {\n\
\treturn jQuery.ajax({\n\
\t\turl: url,\n\
\t\ttype: \"GET\",\n\
\t\tdataType: \"script\",\n\
\t\tasync: false,\n\
\t\tglobal: false,\n\
\t\t\"throws\": true\n\
\t});\n\
};\n\
\n\
\n\
jQuery.fn.extend({\n\
\twrapAll: function( html ) {\n\
\t\tif ( jQuery.isFunction( html ) ) {\n\
\t\t\treturn this.each(function(i) {\n\
\t\t\t\tjQuery(this).wrapAll( html.call(this, i) );\n\
\t\t\t});\n\
\t\t}\n\
\n\
\t\tif ( this[0] ) {\n\
\t\t\t// The elements to wrap the target around\n\
\t\t\tvar wrap = jQuery( html, this[0].ownerDocument ).eq(0).clone(true);\n\
\n\
\t\t\tif ( this[0].parentNode ) {\n\
\t\t\t\twrap.insertBefore( this[0] );\n\
\t\t\t}\n\
\n\
\t\t\twrap.map(function() {\n\
\t\t\t\tvar elem = this;\n\
\n\
\t\t\t\twhile ( elem.firstChild && elem.firstChild.nodeType === 1 ) {\n\
\t\t\t\t\telem = elem.firstChild;\n\
\t\t\t\t}\n\
\n\
\t\t\t\treturn elem;\n\
\t\t\t}).append( this );\n\
\t\t}\n\
\n\
\t\treturn this;\n\
\t},\n\
\n\
\twrapInner: function( html ) {\n\
\t\tif ( jQuery.isFunction( html ) ) {\n\
\t\t\treturn this.each(function(i) {\n\
\t\t\t\tjQuery(this).wrapInner( html.call(this, i) );\n\
\t\t\t});\n\
\t\t}\n\
\n\
\t\treturn this.each(function() {\n\
\t\t\tvar self = jQuery( this ),\n\
\t\t\t\tcontents = self.contents();\n\
\n\
\t\t\tif ( contents.length ) {\n\
\t\t\t\tcontents.wrapAll( html );\n\
\n\
\t\t\t} else {\n\
\t\t\t\tself.append( html );\n\
\t\t\t}\n\
\t\t});\n\
\t},\n\
\n\
\twrap: function( html ) {\n\
\t\tvar isFunction = jQuery.isFunction( html );\n\
\n\
\t\treturn this.each(function(i) {\n\
\t\t\tjQuery( this ).wrapAll( isFunction ? html.call(this, i) : html );\n\
\t\t});\n\
\t},\n\
\n\
\tunwrap: function() {\n\
\t\treturn this.parent().each(function() {\n\
\t\t\tif ( !jQuery.nodeName( this, \"body\" ) ) {\n\
\t\t\t\tjQuery( this ).replaceWith( this.childNodes );\n\
\t\t\t}\n\
\t\t}).end();\n\
\t}\n\
});\n\
\n\
\n\
jQuery.expr.filters.hidden = function( elem ) {\n\
\t// Support: Opera <= 12.12\n\
\t// Opera reports offsetWidths and offsetHeights less than zero on some elements\n\
\treturn elem.offsetWidth <= 0 && elem.offsetHeight <= 0 ||\n\
\t\t(!support.reliableHiddenOffsets() &&\n\
\t\t\t((elem.style && elem.style.display) || jQuery.css( elem, \"display\" )) === \"none\");\n\
};\n\
\n\
jQuery.expr.filters.visible = function( elem ) {\n\
\treturn !jQuery.expr.filters.hidden( elem );\n\
};\n\
\n\
\n\
\n\
\n\
var r20 = /%20/g,\n\
\trbracket = /\\[\\]$/,\n\
\trCRLF = /\\r?\\n\
/g,\n\
\trsubmitterTypes = /^(?:submit|button|image|reset|file)$/i,\n\
\trsubmittable = /^(?:input|select|textarea|keygen)/i;\n\
\n\
function buildParams( prefix, obj, traditional, add ) {\n\
\tvar name;\n\
\n\
\tif ( jQuery.isArray( obj ) ) {\n\
\t\t// Serialize array item.\n\
\t\tjQuery.each( obj, function( i, v ) {\n\
\t\t\tif ( traditional || rbracket.test( prefix ) ) {\n\
\t\t\t\t// Treat each array item as a scalar.\n\
\t\t\t\tadd( prefix, v );\n\
\n\
\t\t\t} else {\n\
\t\t\t\t// Item is non-scalar (array or object), encode its numeric index.\n\
\t\t\t\tbuildParams( prefix + \"[\" + ( typeof v === \"object\" ? i : \"\" ) + \"]\", v, traditional, add );\n\
\t\t\t}\n\
\t\t});\n\
\n\
\t} else if ( !traditional && jQuery.type( obj ) === \"object\" ) {\n\
\t\t// Serialize object item.\n\
\t\tfor ( name in obj ) {\n\
\t\t\tbuildParams( prefix + \"[\" + name + \"]\", obj[ name ], traditional, add );\n\
\t\t}\n\
\n\
\t} else {\n\
\t\t// Serialize scalar item.\n\
\t\tadd( prefix, obj );\n\
\t}\n\
}\n\
\n\
// Serialize an array of form elements or a set of\n\
// key/values into a query string\n\
jQuery.param = function( a, traditional ) {\n\
\tvar prefix,\n\
\t\ts = [],\n\
\t\tadd = function( key, value ) {\n\
\t\t\t// If value is a function, invoke it and return its value\n\
\t\t\tvalue = jQuery.isFunction( value ) ? value() : ( value == null ? \"\" : value );\n\
\t\t\ts[ s.length ] = encodeURIComponent( key ) + \"=\" + encodeURIComponent( value );\n\
\t\t};\n\
\n\
\t// Set traditional to true for jQuery <= 1.3.2 behavior.\n\
\tif ( traditional === undefined ) {\n\
\t\ttraditional = jQuery.ajaxSettings && jQuery.ajaxSettings.traditional;\n\
\t}\n\
\n\
\t// If an array was passed in, assume that it is an array of form elements.\n\
\tif ( jQuery.isArray( a ) || ( a.jquery && !jQuery.isPlainObject( a ) ) ) {\n\
\t\t// Serialize the form elements\n\
\t\tjQuery.each( a, function() {\n\
\t\t\tadd( this.name, this.value );\n\
\t\t});\n\
\n\
\t} else {\n\
\t\t// If traditional, encode the \"old\" way (the way 1.3.2 or older\n\
\t\t// did it), otherwise encode params recursively.\n\
\t\tfor ( prefix in a ) {\n\
\t\t\tbuildParams( prefix, a[ prefix ], traditional, add );\n\
\t\t}\n\
\t}\n\
\n\
\t// Return the resulting serialization\n\
\treturn s.join( \"&\" ).replace( r20, \"+\" );\n\
};\n\
\n\
jQuery.fn.extend({\n\
\tserialize: function() {\n\
\t\treturn jQuery.param( this.serializeArray() );\n\
\t},\n\
\tserializeArray: function() {\n\
\t\treturn this.map(function() {\n\
\t\t\t// Can add propHook for \"elements\" to filter or add form elements\n\
\t\t\tvar elements = jQuery.prop( this, \"elements\" );\n\
\t\t\treturn elements ? jQuery.makeArray( elements ) : this;\n\
\t\t})\n\
\t\t.filter(function() {\n\
\t\t\tvar type = this.type;\n\
\t\t\t// Use .is(\":disabled\") so that fieldset[disabled] works\n\
\t\t\treturn this.name && !jQuery( this ).is( \":disabled\" ) &&\n\
\t\t\t\trsubmittable.test( this.nodeName ) && !rsubmitterTypes.test( type ) &&\n\
\t\t\t\t( this.checked || !rcheckableType.test( type ) );\n\
\t\t})\n\
\t\t.map(function( i, elem ) {\n\
\t\t\tvar val = jQuery( this ).val();\n\
\n\
\t\t\treturn val == null ?\n\
\t\t\t\tnull :\n\
\t\t\t\tjQuery.isArray( val ) ?\n\
\t\t\t\t\tjQuery.map( val, function( val ) {\n\
\t\t\t\t\t\treturn { name: elem.name, value: val.replace( rCRLF, \"\\r\\n\
\" ) };\n\
\t\t\t\t\t}) :\n\
\t\t\t\t\t{ name: elem.name, value: val.replace( rCRLF, \"\\r\\n\
\" ) };\n\
\t\t}).get();\n\
\t}\n\
});\n\
\n\
\n\
// Create the request object\n\
// (This is still attached to ajaxSettings for backward compatibility)\n\
jQuery.ajaxSettings.xhr = window.ActiveXObject !== undefined ?\n\
\t// Support: IE6+\n\
\tfunction() {\n\
\n\
\t\t// XHR cannot access local files, always use ActiveX for that case\n\
\t\treturn !this.isLocal &&\n\
\n\
\t\t\t// Support: IE7-8\n\
\t\t\t// oldIE XHR does not support non-RFC2616 methods (#13240)\n\
\t\t\t// See http://msdn.microsoft.com/en-us/library/ie/ms536648(v=vs.85).aspx\n\
\t\t\t// and http://www.w3.org/Protocols/rfc2616/rfc2616-sec9.html#sec9\n\
\t\t\t// Although this check for six methods instead of eight\n\
\t\t\t// since IE also does not support \"trace\" and \"connect\"\n\
\t\t\t/^(get|post|head|put|delete|options)$/i.test( this.type ) &&\n\
\n\
\t\t\tcreateStandardXHR() || createActiveXHR();\n\
\t} :\n\
\t// For all other browsers, use the standard XMLHttpRequest object\n\
\tcreateStandardXHR;\n\
\n\
var xhrId = 0,\n\
\txhrCallbacks = {},\n\
\txhrSupported = jQuery.ajaxSettings.xhr();\n\
\n\
// Support: IE<10\n\
// Open requests must be manually aborted on unload (#5280)\n\
if ( window.ActiveXObject ) {\n\
\tjQuery( window ).on( \"unload\", function() {\n\
\t\tfor ( var key in xhrCallbacks ) {\n\
\t\t\txhrCallbacks[ key ]( undefined, true );\n\
\t\t}\n\
\t});\n\
}\n\
\n\
// Determine support properties\n\
support.cors = !!xhrSupported && ( \"withCredentials\" in xhrSupported );\n\
xhrSupported = support.ajax = !!xhrSupported;\n\
\n\
// Create transport if the browser can provide an xhr\n\
if ( xhrSupported ) {\n\
\n\
\tjQuery.ajaxTransport(function( options ) {\n\
\t\t// Cross domain only allowed if supported through XMLHttpRequest\n\
\t\tif ( !options.crossDomain || support.cors ) {\n\
\n\
\t\t\tvar callback;\n\
\n\
\t\t\treturn {\n\
\t\t\t\tsend: function( headers, complete ) {\n\
\t\t\t\t\tvar i,\n\
\t\t\t\t\t\txhr = options.xhr(),\n\
\t\t\t\t\t\tid = ++xhrId;\n\
\n\
\t\t\t\t\t// Open the socket\n\
\t\t\t\t\txhr.open( options.type, options.url, options.async, options.username, options.password );\n\
\n\
\t\t\t\t\t// Apply custom fields if provided\n\
\t\t\t\t\tif ( options.xhrFields ) {\n\
\t\t\t\t\t\tfor ( i in options.xhrFields ) {\n\
\t\t\t\t\t\t\txhr[ i ] = options.xhrFields[ i ];\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// Override mime type if needed\n\
\t\t\t\t\tif ( options.mimeType && xhr.overrideMimeType ) {\n\
\t\t\t\t\t\txhr.overrideMimeType( options.mimeType );\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// X-Requested-With header\n\
\t\t\t\t\t// For cross-domain requests, seeing as conditions for a preflight are\n\
\t\t\t\t\t// akin to a jigsaw puzzle, we simply never set it to be sure.\n\
\t\t\t\t\t// (it can always be set on a per-request basis or even using ajaxSetup)\n\
\t\t\t\t\t// For same-domain requests, won't change header if already provided.\n\
\t\t\t\t\tif ( !options.crossDomain && !headers[\"X-Requested-With\"] ) {\n\
\t\t\t\t\t\theaders[\"X-Requested-With\"] = \"XMLHttpRequest\";\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// Set headers\n\
\t\t\t\t\tfor ( i in headers ) {\n\
\t\t\t\t\t\t// Support: IE<9\n\
\t\t\t\t\t\t// IE's ActiveXObject throws a 'Type Mismatch' exception when setting\n\
\t\t\t\t\t\t// request header to a null-value.\n\
\t\t\t\t\t\t//\n\
\t\t\t\t\t\t// To keep consistent with other XHR implementations, cast the value\n\
\t\t\t\t\t\t// to string and ignore `undefined`.\n\
\t\t\t\t\t\tif ( headers[ i ] !== undefined ) {\n\
\t\t\t\t\t\t\txhr.setRequestHeader( i, headers[ i ] + \"\" );\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\n\
\t\t\t\t\t// Do send the request\n\
\t\t\t\t\t// This may raise an exception which is actually\n\
\t\t\t\t\t// handled in jQuery.ajax (so no try/catch here)\n\
\t\t\t\t\txhr.send( ( options.hasContent && options.data ) || null );\n\
\n\
\t\t\t\t\t// Listener\n\
\t\t\t\t\tcallback = function( _, isAbort ) {\n\
\t\t\t\t\t\tvar status, statusText, responses;\n\
\n\
\t\t\t\t\t\t// Was never called and is aborted or complete\n\
\t\t\t\t\t\tif ( callback && ( isAbort || xhr.readyState === 4 ) ) {\n\
\t\t\t\t\t\t\t// Clean up\n\
\t\t\t\t\t\t\tdelete xhrCallbacks[ id ];\n\
\t\t\t\t\t\t\tcallback = undefined;\n\
\t\t\t\t\t\t\txhr.onreadystatechange = jQuery.noop;\n\
\n\
\t\t\t\t\t\t\t// Abort manually if needed\n\
\t\t\t\t\t\t\tif ( isAbort ) {\n\
\t\t\t\t\t\t\t\tif ( xhr.readyState !== 4 ) {\n\
\t\t\t\t\t\t\t\t\txhr.abort();\n\
\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t} else {\n\
\t\t\t\t\t\t\t\tresponses = {};\n\
\t\t\t\t\t\t\t\tstatus = xhr.status;\n\
\n\
\t\t\t\t\t\t\t\t// Support: IE<10\n\
\t\t\t\t\t\t\t\t// Accessing binary-data responseText throws an exception\n\
\t\t\t\t\t\t\t\t// (#11426)\n\
\t\t\t\t\t\t\t\tif ( typeof xhr.responseText === \"string\" ) {\n\
\t\t\t\t\t\t\t\t\tresponses.text = xhr.responseText;\n\
\t\t\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t\t\t\t// Firefox throws an exception when accessing\n\
\t\t\t\t\t\t\t\t// statusText for faulty cross-domain requests\n\
\t\t\t\t\t\t\t\ttry {\n\
\t\t\t\t\t\t\t\t\tstatusText = xhr.statusText;\n\
\t\t\t\t\t\t\t\t} catch( e ) {\n\
\t\t\t\t\t\t\t\t\t// We normalize with Webkit giving an empty statusText\n\
\t\t\t\t\t\t\t\t\tstatusText = \"\";\n\
\t\t\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t\t\t\t// Filter status for non standard behaviors\n\
\n\
\t\t\t\t\t\t\t\t// If the request is local and we have data: assume a success\n\
\t\t\t\t\t\t\t\t// (success with no data won't get notified, that's the best we\n\
\t\t\t\t\t\t\t\t// can do given current implementations)\n\
\t\t\t\t\t\t\t\tif ( !status && options.isLocal && !options.crossDomain ) {\n\
\t\t\t\t\t\t\t\t\tstatus = responses.text ? 200 : 404;\n\
\t\t\t\t\t\t\t\t// IE - #1450: sometimes returns 1223 when it should be 204\n\
\t\t\t\t\t\t\t\t} else if ( status === 1223 ) {\n\
\t\t\t\t\t\t\t\t\tstatus = 204;\n\
\t\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t\t}\n\
\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t\t// Call complete if needed\n\
\t\t\t\t\t\tif ( responses ) {\n\
\t\t\t\t\t\t\tcomplete( status, statusText, responses, xhr.getAllResponseHeaders() );\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t};\n\
\n\
\t\t\t\t\tif ( !options.async ) {\n\
\t\t\t\t\t\t// if we're in sync mode we fire the callback\n\
\t\t\t\t\t\tcallback();\n\
\t\t\t\t\t} else if ( xhr.readyState === 4 ) {\n\
\t\t\t\t\t\t// (IE6 & IE7) if it's in cache and has been\n\
\t\t\t\t\t\t// retrieved directly we need to fire the callback\n\
\t\t\t\t\t\tsetTimeout( callback );\n\
\t\t\t\t\t} else {\n\
\t\t\t\t\t\t// Add to the list of active xhr callbacks\n\
\t\t\t\t\t\txhr.onreadystatechange = xhrCallbacks[ id ] = callback;\n\
\t\t\t\t\t}\n\
\t\t\t\t},\n\
\n\
\t\t\t\tabort: function() {\n\
\t\t\t\t\tif ( callback ) {\n\
\t\t\t\t\t\tcallback( undefined, true );\n\
\t\t\t\t\t}\n\
\t\t\t\t}\n\
\t\t\t};\n\
\t\t}\n\
\t});\n\
}\n\
\n\
// Functions to create xhrs\n\
function createStandardXHR() {\n\
\ttry {\n\
\t\treturn new window.XMLHttpRequest();\n\
\t} catch( e ) {}\n\
}\n\
\n\
function createActiveXHR() {\n\
\ttry {\n\
\t\treturn new window.ActiveXObject( \"Microsoft.XMLHTTP\" );\n\
\t} catch( e ) {}\n\
}\n\
\n\
\n\
\n\
\n\
// Install script dataType\n\
jQuery.ajaxSetup({\n\
\taccepts: {\n\
\t\tscript: \"text/javascript, application/javascript, application/ecmascript, application/x-ecmascript\"\n\
\t},\n\
\tcontents: {\n\
\t\tscript: /(?:java|ecma)script/\n\
\t},\n\
\tconverters: {\n\
\t\t\"text script\": function( text ) {\n\
\t\t\tjQuery.globalEval( text );\n\
\t\t\treturn text;\n\
\t\t}\n\
\t}\n\
});\n\
\n\
// Handle cache's special case and global\n\
jQuery.ajaxPrefilter( \"script\", function( s ) {\n\
\tif ( s.cache === undefined ) {\n\
\t\ts.cache = false;\n\
\t}\n\
\tif ( s.crossDomain ) {\n\
\t\ts.type = \"GET\";\n\
\t\ts.global = false;\n\
\t}\n\
});\n\
\n\
// Bind script tag hack transport\n\
jQuery.ajaxTransport( \"script\", function(s) {\n\
\n\
\t// This transport only deals with cross domain requests\n\
\tif ( s.crossDomain ) {\n\
\n\
\t\tvar script,\n\
\t\t\thead = document.head || jQuery(\"head\")[0] || document.documentElement;\n\
\n\
\t\treturn {\n\
\n\
\t\t\tsend: function( _, callback ) {\n\
\n\
\t\t\t\tscript = document.createElement(\"script\");\n\
\n\
\t\t\t\tscript.async = true;\n\
\n\
\t\t\t\tif ( s.scriptCharset ) {\n\
\t\t\t\t\tscript.charset = s.scriptCharset;\n\
\t\t\t\t}\n\
\n\
\t\t\t\tscript.src = s.url;\n\
\n\
\t\t\t\t// Attach handlers for all browsers\n\
\t\t\t\tscript.onload = script.onreadystatechange = function( _, isAbort ) {\n\
\n\
\t\t\t\t\tif ( isAbort || !script.readyState || /loaded|complete/.test( script.readyState ) ) {\n\
\n\
\t\t\t\t\t\t// Handle memory leak in IE\n\
\t\t\t\t\t\tscript.onload = script.onreadystatechange = null;\n\
\n\
\t\t\t\t\t\t// Remove the script\n\
\t\t\t\t\t\tif ( script.parentNode ) {\n\
\t\t\t\t\t\t\tscript.parentNode.removeChild( script );\n\
\t\t\t\t\t\t}\n\
\n\
\t\t\t\t\t\t// Dereference the script\n\
\t\t\t\t\t\tscript = null;\n\
\n\
\t\t\t\t\t\t// Callback if not abort\n\
\t\t\t\t\t\tif ( !isAbort ) {\n\
\t\t\t\t\t\t\tcallback( 200, \"success\" );\n\
\t\t\t\t\t\t}\n\
\t\t\t\t\t}\n\
\t\t\t\t};\n\
\n\
\t\t\t\t// Circumvent IE6 bugs with base elements (#2709 and #4378) by prepending\n\
\t\t\t\t// Use native DOM manipulation to avoid our domManip AJAX trickery\n\
\t\t\t\thead.insertBefore( script, head.firstChild );\n\
\t\t\t},\n\
\n\
\t\t\tabort: function() {\n\
\t\t\t\tif ( script ) {\n\
\t\t\t\t\tscript.onload( undefined, true );\n\
\t\t\t\t}\n\
\t\t\t}\n\
\t\t};\n\
\t}\n\
});\n\
\n\
\n\
\n\
\n\
var oldCallbacks = [],\n\
\trjsonp = /(=)\\?(?=&|$)|\\?\\?/;\n\
\n\
// Default jsonp settings\n\
jQuery.ajaxSetup({\n\
\tjsonp: \"callback\",\n\
\tjsonpCallback: function() {\n\
\t\tvar callback = oldCallbacks.pop() || ( jQuery.expando + \"_\" + ( nonce++ ) );\n\
\t\tthis[ callback ] = true;\n\
\t\treturn callback;\n\
\t}\n\
});\n\
\n\
// Detect, normalize options and install callbacks for jsonp requests\n\
jQuery.ajaxPrefilter( \"json jsonp\", function( s, originalSettings, jqXHR ) {\n\
\n\
\tvar callbackName, overwritten, responseContainer,\n\
\t\tjsonProp = s.jsonp !== false && ( rjsonp.test( s.url ) ?\n\
\t\t\t\"url\" :\n\
\t\t\ttypeof s.data === \"string\" && !( s.contentType || \"\" ).indexOf(\"application/x-www-form-urlencoded\") && rjsonp.test( s.data ) && \"data\"\n\
\t\t);\n\
\n\
\t// Handle iff the expected data type is \"jsonp\" or we have a parameter to set\n\
\tif ( jsonProp || s.dataTypes[ 0 ] === \"jsonp\" ) {\n\
\n\
\t\t// Get callback name, remembering preexisting value associated with it\n\
\t\tcallbackName = s.jsonpCallback = jQuery.isFunction( s.jsonpCallback ) ?\n\
\t\t\ts.jsonpCallback() :\n\
\t\t\ts.jsonpCallback;\n\
\n\
\t\t// Insert callback into url or form data\n\
\t\tif ( jsonProp ) {\n\
\t\t\ts[ jsonProp ] = s[ jsonProp ].replace( rjsonp, \"$1\" + callbackName );\n\
\t\t} else if ( s.jsonp !== false ) {\n\
\t\t\ts.url += ( rquery.test( s.url ) ? \"&\" : \"?\" ) + s.jsonp + \"=\" + callbackName;\n\
\t\t}\n\
\n\
\t\t// Use data converter to retrieve json after script execution\n\
\t\ts.converters[\"script json\"] = function() {\n\
\t\t\tif ( !responseContainer ) {\n\
\t\t\t\tjQuery.error( callbackName + \" was not called\" );\n\
\t\t\t}\n\
\t\t\treturn responseContainer[ 0 ];\n\
\t\t};\n\
\n\
\t\t// force json dataType\n\
\t\ts.dataTypes[ 0 ] = \"json\";\n\
\n\
\t\t// Install callback\n\
\t\toverwritten = window[ callbackName ];\n\
\t\twindow[ callbackName ] = function() {\n\
\t\t\tresponseContainer = arguments;\n\
\t\t};\n\
\n\
\t\t// Clean-up function (fires after converters)\n\
\t\tjqXHR.always(function() {\n\
\t\t\t// Restore preexisting value\n\
\t\t\twindow[ callbackName ] = overwritten;\n\
\n\
\t\t\t// Save back as free\n\
\t\t\tif ( s[ callbackName ] ) {\n\
\t\t\t\t// make sure that re-using the options doesn't screw things around\n\
\t\t\t\ts.jsonpCallback = originalSettings.jsonpCallback;\n\
\n\
\t\t\t\t// save the callback name for future use\n\
\t\t\t\toldCallbacks.push( callbackName );\n\
\t\t\t}\n\
\n\
\t\t\t// Call if it was a function and we have a response\n\
\t\t\tif ( responseContainer && jQuery.isFunction( overwritten ) ) {\n\
\t\t\t\toverwritten( responseContainer[ 0 ] );\n\
\t\t\t}\n\
\n\
\t\t\tresponseContainer = overwritten = undefined;\n\
\t\t});\n\
\n\
\t\t// Delegate to script\n\
\t\treturn \"script\";\n\
\t}\n\
});\n\
\n\
\n\
\n\
\n\
// data: string of html\n\
// context (optional): If specified, the fragment will be created in this context, defaults to document\n\
// keepScripts (optional): If true, will include scripts passed in the html string\n\
jQuery.parseHTML = function( data, context, keepScripts ) {\n\
\tif ( !data || typeof data !== \"string\" ) {\n\
\t\treturn null;\n\
\t}\n\
\tif ( typeof context === \"boolean\" ) {\n\
\t\tkeepScripts = context;\n\
\t\tcontext = false;\n\
\t}\n\
\tcontext = context || document;\n\
\n\
\tvar parsed = rsingleTag.exec( data ),\n\
\t\tscripts = !keepScripts && [];\n\
\n\
\t// Single tag\n\
\tif ( parsed ) {\n\
\t\treturn [ context.createElement( parsed[1] ) ];\n\
\t}\n\
\n\
\tparsed = jQuery.buildFragment( [ data ], context, scripts );\n\
\n\
\tif ( scripts && scripts.length ) {\n\
\t\tjQuery( scripts ).remove();\n\
\t}\n\
\n\
\treturn jQuery.merge( [], parsed.childNodes );\n\
};\n\
\n\
\n\
// Keep a copy of the old load method\n\
var _load = jQuery.fn.load;\n\
\n\
/**\n\
 * Load a url into a page\n\
 */\n\
jQuery.fn.load = function( url, params, callback ) {\n\
\tif ( typeof url !== \"string\" && _load ) {\n\
\t\treturn _load.apply( this, arguments );\n\
\t}\n\
\n\
\tvar selector, response, type,\n\
\t\tself = this,\n\
\t\toff = url.indexOf(\" \");\n\
\n\
\tif ( off >= 0 ) {\n\
\t\tselector = jQuery.trim( url.slice( off, url.length ) );\n\
\t\turl = url.slice( 0, off );\n\
\t}\n\
\n\
\t// If it's a function\n\
\tif ( jQuery.isFunction( params ) ) {\n\
\n\
\t\t// We assume that it's the callback\n\
\t\tcallback = params;\n\
\t\tparams = undefined;\n\
\n\
\t// Otherwise, build a param string\n\
\t} else if ( params && typeof params === \"object\" ) {\n\
\t\ttype = \"POST\";\n\
\t}\n\
\n\
\t// If we have elements to modify, make the request\n\
\tif ( self.length > 0 ) {\n\
\t\tjQuery.ajax({\n\
\t\t\turl: url,\n\
\n\
\t\t\t// if \"type\" variable is undefined, then \"GET\" method will be used\n\
\t\t\ttype: type,\n\
\t\t\tdataType: \"html\",\n\
\t\t\tdata: params\n\
\t\t}).done(function( responseText ) {\n\
\n\
\t\t\t// Save response for use in complete callback\n\
\t\t\tresponse = arguments;\n\
\n\
\t\t\tself.html( selector ?\n\
\n\
\t\t\t\t// If a selector was specified, locate the right elements in a dummy div\n\
\t\t\t\t// Exclude scripts to avoid IE 'Permission Denied' errors\n\
\t\t\t\tjQuery(\"<div>\").append( jQuery.parseHTML( responseText ) ).find( selector ) :\n\
\n\
\t\t\t\t// Otherwise use the full result\n\
\t\t\t\tresponseText );\n\
\n\
\t\t}).complete( callback && function( jqXHR, status ) {\n\
\t\t\tself.each( callback, response || [ jqXHR.responseText, status, jqXHR ] );\n\
\t\t});\n\
\t}\n\
\n\
\treturn this;\n\
};\n\
\n\
\n\
\n\
\n\
jQuery.expr.filters.animated = function( elem ) {\n\
\treturn jQuery.grep(jQuery.timers, function( fn ) {\n\
\t\treturn elem === fn.elem;\n\
\t}).length;\n\
};\n\
\n\
\n\
\n\
\n\
\n\
var docElem = window.document.documentElement;\n\
\n\
/**\n\
 * Gets a window from an element\n\
 */\n\
function getWindow( elem ) {\n\
\treturn jQuery.isWindow( elem ) ?\n\
\t\telem :\n\
\t\telem.nodeType === 9 ?\n\
\t\t\telem.defaultView || elem.parentWindow :\n\
\t\t\tfalse;\n\
}\n\
\n\
jQuery.offset = {\n\
\tsetOffset: function( elem, options, i ) {\n\
\t\tvar curPosition, curLeft, curCSSTop, curTop, curOffset, curCSSLeft, calculatePosition,\n\
\t\t\tposition = jQuery.css( elem, \"position\" ),\n\
\t\t\tcurElem = jQuery( elem ),\n\
\t\t\tprops = {};\n\
\n\
\t\t// set position first, in-case top/left are set even on static elem\n\
\t\tif ( position === \"static\" ) {\n\
\t\t\telem.style.position = \"relative\";\n\
\t\t}\n\
\n\
\t\tcurOffset = curElem.offset();\n\
\t\tcurCSSTop = jQuery.css( elem, \"top\" );\n\
\t\tcurCSSLeft = jQuery.css( elem, \"left\" );\n\
\t\tcalculatePosition = ( position === \"absolute\" || position === \"fixed\" ) &&\n\
\t\t\tjQuery.inArray(\"auto\", [ curCSSTop, curCSSLeft ] ) > -1;\n\
\n\
\t\t// need to be able to calculate position if either top or left is auto and position is either absolute or fixed\n\
\t\tif ( calculatePosition ) {\n\
\t\t\tcurPosition = curElem.position();\n\
\t\t\tcurTop = curPosition.top;\n\
\t\t\tcurLeft = curPosition.left;\n\
\t\t} else {\n\
\t\t\tcurTop = parseFloat( curCSSTop ) || 0;\n\
\t\t\tcurLeft = parseFloat( curCSSLeft ) || 0;\n\
\t\t}\n\
\n\
\t\tif ( jQuery.isFunction( options ) ) {\n\
\t\t\toptions = options.call( elem, i, curOffset );\n\
\t\t}\n\
\n\
\t\tif ( options.top != null ) {\n\
\t\t\tprops.top = ( options.top - curOffset.top ) + curTop;\n\
\t\t}\n\
\t\tif ( options.left != null ) {\n\
\t\t\tprops.left = ( options.left - curOffset.left ) + curLeft;\n\
\t\t}\n\
\n\
\t\tif ( \"using\" in options ) {\n\
\t\t\toptions.using.call( elem, props );\n\
\t\t} else {\n\
\t\t\tcurElem.css( props );\n\
\t\t}\n\
\t}\n\
};\n\
\n\
jQuery.fn.extend({\n\
\toffset: function( options ) {\n\
\t\tif ( arguments.length ) {\n\
\t\t\treturn options === undefined ?\n\
\t\t\t\tthis :\n\
\t\t\t\tthis.each(function( i ) {\n\
\t\t\t\t\tjQuery.offset.setOffset( this, options, i );\n\
\t\t\t\t});\n\
\t\t}\n\
\n\
\t\tvar docElem, win,\n\
\t\t\tbox = { top: 0, left: 0 },\n\
\t\t\telem = this[ 0 ],\n\
\t\t\tdoc = elem && elem.ownerDocument;\n\
\n\
\t\tif ( !doc ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\tdocElem = doc.documentElement;\n\
\n\
\t\t// Make sure it's not a disconnected DOM node\n\
\t\tif ( !jQuery.contains( docElem, elem ) ) {\n\
\t\t\treturn box;\n\
\t\t}\n\
\n\
\t\t// If we don't have gBCR, just use 0,0 rather than error\n\
\t\t// BlackBerry 5, iOS 3 (original iPhone)\n\
\t\tif ( typeof elem.getBoundingClientRect !== strundefined ) {\n\
\t\t\tbox = elem.getBoundingClientRect();\n\
\t\t}\n\
\t\twin = getWindow( doc );\n\
\t\treturn {\n\
\t\t\ttop: box.top  + ( win.pageYOffset || docElem.scrollTop )  - ( docElem.clientTop  || 0 ),\n\
\t\t\tleft: box.left + ( win.pageXOffset || docElem.scrollLeft ) - ( docElem.clientLeft || 0 )\n\
\t\t};\n\
\t},\n\
\n\
\tposition: function() {\n\
\t\tif ( !this[ 0 ] ) {\n\
\t\t\treturn;\n\
\t\t}\n\
\n\
\t\tvar offsetParent, offset,\n\
\t\t\tparentOffset = { top: 0, left: 0 },\n\
\t\t\telem = this[ 0 ];\n\
\n\
\t\t// fixed elements are offset from window (parentOffset = {top:0, left: 0}, because it is its only offset parent\n\
\t\tif ( jQuery.css( elem, \"position\" ) === \"fixed\" ) {\n\
\t\t\t// we assume that getBoundingClientRect is available when computed position is fixed\n\
\t\t\toffset = elem.getBoundingClientRect();\n\
\t\t} else {\n\
\t\t\t// Get *real* offsetParent\n\
\t\t\toffsetParent = this.offsetParent();\n\
\n\
\t\t\t// Get correct offsets\n\
\t\t\toffset = this.offset();\n\
\t\t\tif ( !jQuery.nodeName( offsetParent[ 0 ], \"html\" ) ) {\n\
\t\t\t\tparentOffset = offsetParent.offset();\n\
\t\t\t}\n\
\n\
\t\t\t// Add offsetParent borders\n\
\t\t\tparentOffset.top  += jQuery.css( offsetParent[ 0 ], \"borderTopWidth\", true );\n\
\t\t\tparentOffset.left += jQuery.css( offsetParent[ 0 ], \"borderLeftWidth\", true );\n\
\t\t}\n\
\n\
\t\t// Subtract parent offsets and element margins\n\
\t\t// note: when an element has margin: auto the offsetLeft and marginLeft\n\
\t\t// are the same in Safari causing offset.left to incorrectly be 0\n\
\t\treturn {\n\
\t\t\ttop:  offset.top  - parentOffset.top - jQuery.css( elem, \"marginTop\", true ),\n\
\t\t\tleft: offset.left - parentOffset.left - jQuery.css( elem, \"marginLeft\", true)\n\
\t\t};\n\
\t},\n\
\n\
\toffsetParent: function() {\n\
\t\treturn this.map(function() {\n\
\t\t\tvar offsetParent = this.offsetParent || docElem;\n\
\n\
\t\t\twhile ( offsetParent && ( !jQuery.nodeName( offsetParent, \"html\" ) && jQuery.css( offsetParent, \"position\" ) === \"static\" ) ) {\n\
\t\t\t\toffsetParent = offsetParent.offsetParent;\n\
\t\t\t}\n\
\t\t\treturn offsetParent || docElem;\n\
\t\t});\n\
\t}\n\
});\n\
\n\
// Create scrollLeft and scrollTop methods\n\
jQuery.each( { scrollLeft: \"pageXOffset\", scrollTop: \"pageYOffset\" }, function( method, prop ) {\n\
\tvar top = /Y/.test( prop );\n\
\n\
\tjQuery.fn[ method ] = function( val ) {\n\
\t\treturn access( this, function( elem, method, val ) {\n\
\t\t\tvar win = getWindow( elem );\n\
\n\
\t\t\tif ( val === undefined ) {\n\
\t\t\t\treturn win ? (prop in win) ? win[ prop ] :\n\
\t\t\t\t\twin.document.documentElement[ method ] :\n\
\t\t\t\t\telem[ method ];\n\
\t\t\t}\n\
\n\
\t\t\tif ( win ) {\n\
\t\t\t\twin.scrollTo(\n\
\t\t\t\t\t!top ? val : jQuery( win ).scrollLeft(),\n\
\t\t\t\t\ttop ? val : jQuery( win ).scrollTop()\n\
\t\t\t\t);\n\
\n\
\t\t\t} else {\n\
\t\t\t\telem[ method ] = val;\n\
\t\t\t}\n\
\t\t}, method, val, arguments.length, null );\n\
\t};\n\
});\n\
\n\
// Add the top/left cssHooks using jQuery.fn.position\n\
// Webkit bug: https://bugs.webkit.org/show_bug.cgi?id=29084\n\
// getComputedStyle returns percent when specified for top/left/bottom/right\n\
// rather than make the css module depend on the offset module, we just check for it here\n\
jQuery.each( [ \"top\", \"left\" ], function( i, prop ) {\n\
\tjQuery.cssHooks[ prop ] = addGetHookIf( support.pixelPosition,\n\
\t\tfunction( elem, computed ) {\n\
\t\t\tif ( computed ) {\n\
\t\t\t\tcomputed = curCSS( elem, prop );\n\
\t\t\t\t// if curCSS returns percentage, fallback to offset\n\
\t\t\t\treturn rnumnonpx.test( computed ) ?\n\
\t\t\t\t\tjQuery( elem ).position()[ prop ] + \"px\" :\n\
\t\t\t\t\tcomputed;\n\
\t\t\t}\n\
\t\t}\n\
\t);\n\
});\n\
\n\
\n\
// Create innerHeight, innerWidth, height, width, outerHeight and outerWidth methods\n\
jQuery.each( { Height: \"height\", Width: \"width\" }, function( name, type ) {\n\
\tjQuery.each( { padding: \"inner\" + name, content: type, \"\": \"outer\" + name }, function( defaultExtra, funcName ) {\n\
\t\t// margin is only for outerHeight, outerWidth\n\
\t\tjQuery.fn[ funcName ] = function( margin, value ) {\n\
\t\t\tvar chainable = arguments.length && ( defaultExtra || typeof margin !== \"boolean\" ),\n\
\t\t\t\textra = defaultExtra || ( margin === true || value === true ? \"margin\" : \"border\" );\n\
\n\
\t\t\treturn access( this, function( elem, type, value ) {\n\
\t\t\t\tvar doc;\n\
\n\
\t\t\t\tif ( jQuery.isWindow( elem ) ) {\n\
\t\t\t\t\t// As of 5/8/2012 this will yield incorrect results for Mobile Safari, but there\n\
\t\t\t\t\t// isn't a whole lot we can do. See pull request at this URL for discussion:\n\
\t\t\t\t\t// https://github.com/jquery/jquery/pull/764\n\
\t\t\t\t\treturn elem.document.documentElement[ \"client\" + name ];\n\
\t\t\t\t}\n\
\n\
\t\t\t\t// Get document width or height\n\
\t\t\t\tif ( elem.nodeType === 9 ) {\n\
\t\t\t\t\tdoc = elem.documentElement;\n\
\n\
\t\t\t\t\t// Either scroll[Width/Height] or offset[Width/Height] or client[Width/Height], whichever is greatest\n\
\t\t\t\t\t// unfortunately, this causes bug #3838 in IE6/8 only, but there is currently no good, small way to fix it.\n\
\t\t\t\t\treturn Math.max(\n\
\t\t\t\t\t\telem.body[ \"scroll\" + name ], doc[ \"scroll\" + name ],\n\
\t\t\t\t\t\telem.body[ \"offset\" + name ], doc[ \"offset\" + name ],\n\
\t\t\t\t\t\tdoc[ \"client\" + name ]\n\
\t\t\t\t\t);\n\
\t\t\t\t}\n\
\n\
\t\t\t\treturn value === undefined ?\n\
\t\t\t\t\t// Get width or height on the element, requesting but not forcing parseFloat\n\
\t\t\t\t\tjQuery.css( elem, type, extra ) :\n\
\n\
\t\t\t\t\t// Set width or height on the element\n\
\t\t\t\t\tjQuery.style( elem, type, value, extra );\n\
\t\t\t}, type, chainable ? margin : undefined, chainable, null );\n\
\t\t};\n\
\t});\n\
});\n\
\n\
\n\
// The number of elements contained in the matched element set\n\
jQuery.fn.size = function() {\n\
\treturn this.length;\n\
};\n\
\n\
jQuery.fn.andSelf = jQuery.fn.addBack;\n\
\n\
\n\
\n\
\n\
// Register as a named AMD module, since jQuery can be concatenated with other\n\
// files that may use define, but not via a proper concatenation script that\n\
// understands anonymous AMD modules. A named AMD is safest and most robust\n\
// way to register. Lowercase jquery is used because AMD module names are\n\
// derived from file names, and jQuery is normally delivered in a lowercase\n\
// file name. Do this after creating the global so that if an AMD module wants\n\
// to call noConflict to hide this version of jQuery, it will work.\n\
\n\
// Note that for maximum portability, libraries that are not jQuery should\n\
// declare themselves as anonymous modules, and avoid setting a global if an\n\
// AMD loader is present. jQuery is a special case. For more information, see\n\
// https://github.com/jrburke/requirejs/wiki/Updating-existing-libraries#wiki-anon\n\
\n\
if ( typeof define === \"function\" && define.amd ) {\n\
\tdefine( \"jquery\", [], function() {\n\
\t\treturn jQuery;\n\
\t});\n\
}\n\
\n\
\n\
\n\
\n\
var\n\
\t// Map over jQuery in case of overwrite\n\
\t_jQuery = window.jQuery,\n\
\n\
\t// Map over the $ in case of overwrite\n\
\t_$ = window.$;\n\
\n\
jQuery.noConflict = function( deep ) {\n\
\tif ( window.$ === jQuery ) {\n\
\t\twindow.$ = _$;\n\
\t}\n\
\n\
\tif ( deep && window.jQuery === jQuery ) {\n\
\t\twindow.jQuery = _jQuery;\n\
\t}\n\
\n\
\treturn jQuery;\n\
};\n\
\n\
// Expose jQuery and $ identifiers, even in\n\
// AMD (#7102#comment:10, https://github.com/jquery/jquery/pull/557)\n\
// and CommonJS for browser emulators (#13566)\n\
if ( typeof noGlobal === strundefined ) {\n\
\twindow.jQuery = window.$ = jQuery;\n\
}\n\
\n\
\n\
\n\
\n\
return jQuery;\n\
\n\
}));";

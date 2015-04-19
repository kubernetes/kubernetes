/* Modernizr 2.0.6 (Custom Build) | MIT & BSD
 * Build: http://www.modernizr.com/download/#-iepp
 */
;window.Modernizr=function(a,b,c){function w(a,b){return!!~(""+a).indexOf(b)}function v(a,b){return typeof a===b}function u(a,b){return t(prefixes.join(a+";")+(b||""))}function t(a){j.cssText=a}var d="2.0.6",e={},f=b.documentElement,g=b.head||b.getElementsByTagName("head")[0],h="modernizr",i=b.createElement(h),j=i.style,k,l=Object.prototype.toString,m={},n={},o={},p=[],q,r={}.hasOwnProperty,s;!v(r,c)&&!v(r.call,c)?s=function(a,b){return r.call(a,b)}:s=function(a,b){return b in a&&v(a.constructor.prototype[b],c)};for(var x in m)s(m,x)&&(q=x.toLowerCase(),e[q]=m[x](),p.push((e[q]?"":"no-")+q));t(""),i=k=null,a.attachEvent&&function(){var a=b.createElement("div");a.innerHTML="<elem></elem>";return a.childNodes.length!==1}()&&function(a,b){function s(a){var b=-1;while(++b<g)a.createElement(f[b])}a.iepp=a.iepp||{};var d=a.iepp,e=d.html5elements||"abbr|article|aside|audio|canvas|datalist|details|figcaption|figure|footer|header|hgroup|mark|meter|nav|output|progress|section|summary|time|video",f=e.split("|"),g=f.length,h=new RegExp("(^|\\s)("+e+")","gi"),i=new RegExp("<(/*)("+e+")","gi"),j=/^\s*[\{\}]\s*$/,k=new RegExp("(^|[^\\n]*?\\s)("+e+")([^\\n]*)({[\\n\\w\\W]*?})","gi"),l=b.createDocumentFragment(),m=b.documentElement,n=m.firstChild,o=b.createElement("body"),p=b.createElement("style"),q=/print|all/,r;d.getCSS=function(a,b){if(a+""===c)return"";var e=-1,f=a.length,g,h=[];while(++e<f){g=a[e];if(g.disabled)continue;b=g.media||b,q.test(b)&&h.push(d.getCSS(g.imports,b),g.cssText),b="all"}return h.join("")},d.parseCSS=function(a){var b=[],c;while((c=k.exec(a))!=null)b.push(((j.exec(c[1])?"\n":c[1])+c[2]+c[3]).replace(h,"$1.iepp_$2")+c[4]);return b.join("\n")},d.writeHTML=function(){var a=-1;r=r||b.body;while(++a<g){var c=b.getElementsByTagName(f[a]),d=c.length,e=-1;while(++e<d)c[e].className.indexOf("iepp_")<0&&(c[e].className+=" iepp_"+f[a])}l.appendChild(r),m.appendChild(o),o.className=r.className,o.id=r.id,o.innerHTML=r.innerHTML.replace(i,"<$1font")},d._beforePrint=function(){p.styleSheet.cssText=d.parseCSS(d.getCSS(b.styleSheets,"all")),d.writeHTML()},d.restoreHTML=function(){o.innerHTML="",m.removeChild(o),m.appendChild(r)},d._afterPrint=function(){d.restoreHTML(),p.styleSheet.cssText=""},s(b),s(l);d.disablePP||(n.insertBefore(p,n.firstChild),p.media="print",p.className="iepp-printshim",a.attachEvent("onbeforeprint",d._beforePrint),a.attachEvent("onafterprint",d._afterPrint))}(a,b),e._version=d;return e}(this,this.document);
(function (con) {
    // the dummy function
    function dummy() {};
    // console methods that may exist
    for(var methods = "assert,count,debug,dir,dirxml,error,exception,group,groupCollapsed,groupEnd,info,log,markTimeline,profile,profileEnd,time,timeEnd,trace,warn".split(','), func; func = methods.pop();) {
        con[func] = con[func] || dummy;
    }
}(window.console = window.console || {})); 
// we do this crazy little dance so that the `console` object
// inside the function is a name that can be shortened to a single
// letter by the compressor to make the compressed script as tiny
// as possible.
/*!
 * jQuery JavaScript Library v1.6.3
 * http://jquery.com/
 *
 * Copyright 2011, John Resig
 * Dual licensed under the MIT or GPL Version 2 licenses.
 * http://jquery.org/license
 *
 * Includes Sizzle.js
 * http://sizzlejs.com/
 * Copyright 2011, The Dojo Foundation
 * Released under the MIT, BSD, and GPL Licenses.
 *
 * Date: Wed Aug 31 10:35:15 2011 -0400
 */
(function( window, undefined ) {

// Use the correct document accordingly with window argument (sandbox)
var document = window.document,
	navigator = window.navigator,
	location = window.location;
var jQuery = (function() {

// Define a local copy of jQuery
var jQuery = function( selector, context ) {
		// The jQuery object is actually just the init constructor 'enhanced'
		return new jQuery.fn.init( selector, context, rootjQuery );
	},

	// Map over jQuery in case of overwrite
	_jQuery = window.jQuery,

	// Map over the $ in case of overwrite
	_$ = window.$,

	// A central reference to the root jQuery(document)
	rootjQuery,

	// A simple way to check for HTML strings or ID strings
	// Prioritize #id over <tag> to avoid XSS via location.hash (#9521)
	quickExpr = /^(?:[^#<]*(<[\w\W]+>)[^>]*$|#([\w\-]*)$)/,

	// Check if a string has a non-whitespace character in it
	rnotwhite = /\S/,

	// Used for trimming whitespace
	trimLeft = /^\s+/,
	trimRight = /\s+$/,

	// Check for digits
	rdigit = /\d/,

	// Match a standalone tag
	rsingleTag = /^<(\w+)\s*\/?>(?:<\/\1>)?$/,

	// JSON RegExp
	rvalidchars = /^[\],:{}\s]*$/,
	rvalidescape = /\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g,
	rvalidtokens = /"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g,
	rvalidbraces = /(?:^|:|,)(?:\s*\[)+/g,

	// Useragent RegExp
	rwebkit = /(webkit)[ \/]([\w.]+)/,
	ropera = /(opera)(?:.*version)?[ \/]([\w.]+)/,
	rmsie = /(msie) ([\w.]+)/,
	rmozilla = /(mozilla)(?:.*? rv:([\w.]+))?/,

	// Matches dashed string for camelizing
	rdashAlpha = /-([a-z]|[0-9])/ig,
	rmsPrefix = /^-ms-/,

	// Used by jQuery.camelCase as callback to replace()
	fcamelCase = function( all, letter ) {
		return ( letter + "" ).toUpperCase();
	},

	// Keep a UserAgent string for use with jQuery.browser
	userAgent = navigator.userAgent,

	// For matching the engine and version of the browser
	browserMatch,

	// The deferred used on DOM ready
	readyList,

	// The ready event handler
	DOMContentLoaded,

	// Save a reference to some core methods
	toString = Object.prototype.toString,
	hasOwn = Object.prototype.hasOwnProperty,
	push = Array.prototype.push,
	slice = Array.prototype.slice,
	trim = String.prototype.trim,
	indexOf = Array.prototype.indexOf,

	// [[Class]] -> type pairs
	class2type = {};

jQuery.fn = jQuery.prototype = {
	constructor: jQuery,
	init: function( selector, context, rootjQuery ) {
		var match, elem, ret, doc;

		// Handle $(""), $(null), or $(undefined)
		if ( !selector ) {
			return this;
		}

		// Handle $(DOMElement)
		if ( selector.nodeType ) {
			this.context = this[0] = selector;
			this.length = 1;
			return this;
		}

		// The body element only exists once, optimize finding it
		if ( selector === "body" && !context && document.body ) {
			this.context = document;
			this[0] = document.body;
			this.selector = selector;
			this.length = 1;
			return this;
		}

		// Handle HTML strings
		if ( typeof selector === "string" ) {
			// Are we dealing with HTML string or an ID?
			if ( selector.charAt(0) === "<" && selector.charAt( selector.length - 1 ) === ">" && selector.length >= 3 ) {
				// Assume that strings that start and end with <> are HTML and skip the regex check
				match = [ null, selector, null ];

			} else {
				match = quickExpr.exec( selector );
			}

			// Verify a match, and that no context was specified for #id
			if ( match && (match[1] || !context) ) {

				// HANDLE: $(html) -> $(array)
				if ( match[1] ) {
					context = context instanceof jQuery ? context[0] : context;
					doc = (context ? context.ownerDocument || context : document);

					// If a single string is passed in and it's a single tag
					// just do a createElement and skip the rest
					ret = rsingleTag.exec( selector );

					if ( ret ) {
						if ( jQuery.isPlainObject( context ) ) {
							selector = [ document.createElement( ret[1] ) ];
							jQuery.fn.attr.call( selector, context, true );

						} else {
							selector = [ doc.createElement( ret[1] ) ];
						}

					} else {
						ret = jQuery.buildFragment( [ match[1] ], [ doc ] );
						selector = (ret.cacheable ? jQuery.clone(ret.fragment) : ret.fragment).childNodes;
					}

					return jQuery.merge( this, selector );

				// HANDLE: $("#id")
				} else {
					elem = document.getElementById( match[2] );

					// Check parentNode to catch when Blackberry 4.6 returns
					// nodes that are no longer in the document #6963
					if ( elem && elem.parentNode ) {
						// Handle the case where IE and Opera return items
						// by name instead of ID
						if ( elem.id !== match[2] ) {
							return rootjQuery.find( selector );
						}

						// Otherwise, we inject the element directly into the jQuery object
						this.length = 1;
						this[0] = elem;
					}

					this.context = document;
					this.selector = selector;
					return this;
				}

			// HANDLE: $(expr, $(...))
			} else if ( !context || context.jquery ) {
				return (context || rootjQuery).find( selector );

			// HANDLE: $(expr, context)
			// (which is just equivalent to: $(context).find(expr)
			} else {
				return this.constructor( context ).find( selector );
			}

		// HANDLE: $(function)
		// Shortcut for document ready
		} else if ( jQuery.isFunction( selector ) ) {
			return rootjQuery.ready( selector );
		}

		if (selector.selector !== undefined) {
			this.selector = selector.selector;
			this.context = selector.context;
		}

		return jQuery.makeArray( selector, this );
	},

	// Start with an empty selector
	selector: "",

	// The current version of jQuery being used
	jquery: "1.6.3",

	// The default length of a jQuery object is 0
	length: 0,

	// The number of elements contained in the matched element set
	size: function() {
		return this.length;
	},

	toArray: function() {
		return slice.call( this, 0 );
	},

	// Get the Nth element in the matched element set OR
	// Get the whole matched element set as a clean array
	get: function( num ) {
		return num == null ?

			// Return a 'clean' array
			this.toArray() :

			// Return just the object
			( num < 0 ? this[ this.length + num ] : this[ num ] );
	},

	// Take an array of elements and push it onto the stack
	// (returning the new matched element set)
	pushStack: function( elems, name, selector ) {
		// Build a new jQuery matched element set
		var ret = this.constructor();

		if ( jQuery.isArray( elems ) ) {
			push.apply( ret, elems );

		} else {
			jQuery.merge( ret, elems );
		}

		// Add the old object onto the stack (as a reference)
		ret.prevObject = this;

		ret.context = this.context;

		if ( name === "find" ) {
			ret.selector = this.selector + (this.selector ? " " : "") + selector;
		} else if ( name ) {
			ret.selector = this.selector + "." + name + "(" + selector + ")";
		}

		// Return the newly-formed element set
		return ret;
	},

	// Execute a callback for every element in the matched set.
	// (You can seed the arguments with an array of args, but this is
	// only used internally.)
	each: function( callback, args ) {
		return jQuery.each( this, callback, args );
	},

	ready: function( fn ) {
		// Attach the listeners
		jQuery.bindReady();

		// Add the callback
		readyList.done( fn );

		return this;
	},

	eq: function( i ) {
		return i === -1 ?
			this.slice( i ) :
			this.slice( i, +i + 1 );
	},

	first: function() {
		return this.eq( 0 );
	},

	last: function() {
		return this.eq( -1 );
	},

	slice: function() {
		return this.pushStack( slice.apply( this, arguments ),
			"slice", slice.call(arguments).join(",") );
	},

	map: function( callback ) {
		return this.pushStack( jQuery.map(this, function( elem, i ) {
			return callback.call( elem, i, elem );
		}));
	},

	end: function() {
		return this.prevObject || this.constructor(null);
	},

	// For internal use only.
	// Behaves like an Array's method, not like a jQuery method.
	push: push,
	sort: [].sort,
	splice: [].splice
};

// Give the init function the jQuery prototype for later instantiation
jQuery.fn.init.prototype = jQuery.fn;

jQuery.extend = jQuery.fn.extend = function() {
	var options, name, src, copy, copyIsArray, clone,
		target = arguments[0] || {},
		i = 1,
		length = arguments.length,
		deep = false;

	// Handle a deep copy situation
	if ( typeof target === "boolean" ) {
		deep = target;
		target = arguments[1] || {};
		// skip the boolean and the target
		i = 2;
	}

	// Handle case when target is a string or something (possible in deep copy)
	if ( typeof target !== "object" && !jQuery.isFunction(target) ) {
		target = {};
	}

	// extend jQuery itself if only one argument is passed
	if ( length === i ) {
		target = this;
		--i;
	}

	for ( ; i < length; i++ ) {
		// Only deal with non-null/undefined values
		if ( (options = arguments[ i ]) != null ) {
			// Extend the base object
			for ( name in options ) {
				src = target[ name ];
				copy = options[ name ];

				// Prevent never-ending loop
				if ( target === copy ) {
					continue;
				}

				// Recurse if we're merging plain objects or arrays
				if ( deep && copy && ( jQuery.isPlainObject(copy) || (copyIsArray = jQuery.isArray(copy)) ) ) {
					if ( copyIsArray ) {
						copyIsArray = false;
						clone = src && jQuery.isArray(src) ? src : [];

					} else {
						clone = src && jQuery.isPlainObject(src) ? src : {};
					}

					// Never move original objects, clone them
					target[ name ] = jQuery.extend( deep, clone, copy );

				// Don't bring in undefined values
				} else if ( copy !== undefined ) {
					target[ name ] = copy;
				}
			}
		}
	}

	// Return the modified object
	return target;
};

jQuery.extend({
	noConflict: function( deep ) {
		if ( window.$ === jQuery ) {
			window.$ = _$;
		}

		if ( deep && window.jQuery === jQuery ) {
			window.jQuery = _jQuery;
		}

		return jQuery;
	},

	// Is the DOM ready to be used? Set to true once it occurs.
	isReady: false,

	// A counter to track how many items to wait for before
	// the ready event fires. See #6781
	readyWait: 1,

	// Hold (or release) the ready event
	holdReady: function( hold ) {
		if ( hold ) {
			jQuery.readyWait++;
		} else {
			jQuery.ready( true );
		}
	},

	// Handle when the DOM is ready
	ready: function( wait ) {
		// Either a released hold or an DOMready/load event and not yet ready
		if ( (wait === true && !--jQuery.readyWait) || (wait !== true && !jQuery.isReady) ) {
			// Make sure body exists, at least, in case IE gets a little overzealous (ticket #5443).
			if ( !document.body ) {
				return setTimeout( jQuery.ready, 1 );
			}

			// Remember that the DOM is ready
			jQuery.isReady = true;

			// If a normal DOM Ready event fired, decrement, and wait if need be
			if ( wait !== true && --jQuery.readyWait > 0 ) {
				return;
			}

			// If there are functions bound, to execute
			readyList.resolveWith( document, [ jQuery ] );

			// Trigger any bound ready events
			if ( jQuery.fn.trigger ) {
				jQuery( document ).trigger( "ready" ).unbind( "ready" );
			}
		}
	},

	bindReady: function() {
		if ( readyList ) {
			return;
		}

		readyList = jQuery._Deferred();

		// Catch cases where $(document).ready() is called after the
		// browser event has already occurred.
		if ( document.readyState === "complete" ) {
			// Handle it asynchronously to allow scripts the opportunity to delay ready
			return setTimeout( jQuery.ready, 1 );
		}

		// Mozilla, Opera and webkit nightlies currently support this event
		if ( document.addEventListener ) {
			// Use the handy event callback
			document.addEventListener( "DOMContentLoaded", DOMContentLoaded, false );

			// A fallback to window.onload, that will always work
			window.addEventListener( "load", jQuery.ready, false );

		// If IE event model is used
		} else if ( document.attachEvent ) {
			// ensure firing before onload,
			// maybe late but safe also for iframes
			document.attachEvent( "onreadystatechange", DOMContentLoaded );

			// A fallback to window.onload, that will always work
			window.attachEvent( "onload", jQuery.ready );

			// If IE and not a frame
			// continually check to see if the document is ready
			var toplevel = false;

			try {
				toplevel = window.frameElement == null;
			} catch(e) {}

			if ( document.documentElement.doScroll && toplevel ) {
				doScrollCheck();
			}
		}
	},

	// See test/unit/core.js for details concerning isFunction.
	// Since version 1.3, DOM methods and functions like alert
	// aren't supported. They return false on IE (#2968).
	isFunction: function( obj ) {
		return jQuery.type(obj) === "function";
	},

	isArray: Array.isArray || function( obj ) {
		return jQuery.type(obj) === "array";
	},

	// A crude way of determining if an object is a window
	isWindow: function( obj ) {
		return obj && typeof obj === "object" && "setInterval" in obj;
	},

	isNaN: function( obj ) {
		return obj == null || !rdigit.test( obj ) || isNaN( obj );
	},

	type: function( obj ) {
		return obj == null ?
			String( obj ) :
			class2type[ toString.call(obj) ] || "object";
	},

	isPlainObject: function( obj ) {
		// Must be an Object.
		// Because of IE, we also have to check the presence of the constructor property.
		// Make sure that DOM nodes and window objects don't pass through, as well
		if ( !obj || jQuery.type(obj) !== "object" || obj.nodeType || jQuery.isWindow( obj ) ) {
			return false;
		}

		try {
			// Not own constructor property must be Object
			if ( obj.constructor &&
				!hasOwn.call(obj, "constructor") &&
				!hasOwn.call(obj.constructor.prototype, "isPrototypeOf") ) {
				return false;
			}
		} catch ( e ) {
			// IE8,9 Will throw exceptions on certain host objects #9897
			return false;
		}

		// Own properties are enumerated firstly, so to speed up,
		// if last one is own, then all properties are own.

		var key;
		for ( key in obj ) {}

		return key === undefined || hasOwn.call( obj, key );
	},

	isEmptyObject: function( obj ) {
		for ( var name in obj ) {
			return false;
		}
		return true;
	},

	error: function( msg ) {
		throw msg;
	},

	parseJSON: function( data ) {
		if ( typeof data !== "string" || !data ) {
			return null;
		}

		// Make sure leading/trailing whitespace is removed (IE can't handle it)
		data = jQuery.trim( data );

		// Attempt to parse using the native JSON parser first
		if ( window.JSON && window.JSON.parse ) {
			return window.JSON.parse( data );
		}

		// Make sure the incoming data is actual JSON
		// Logic borrowed from http://json.org/json2.js
		if ( rvalidchars.test( data.replace( rvalidescape, "@" )
			.replace( rvalidtokens, "]" )
			.replace( rvalidbraces, "")) ) {

			return (new Function( "return " + data ))();

		}
		jQuery.error( "Invalid JSON: " + data );
	},

	// Cross-browser xml parsing
	parseXML: function( data ) {
		var xml, tmp;
		try {
			if ( window.DOMParser ) { // Standard
				tmp = new DOMParser();
				xml = tmp.parseFromString( data , "text/xml" );
			} else { // IE
				xml = new ActiveXObject( "Microsoft.XMLDOM" );
				xml.async = "false";
				xml.loadXML( data );
			}
		} catch( e ) {
			xml = undefined;
		}
		if ( !xml || !xml.documentElement || xml.getElementsByTagName( "parsererror" ).length ) {
			jQuery.error( "Invalid XML: " + data );
		}
		return xml;
	},

	noop: function() {},

	// Evaluates a script in a global context
	// Workarounds based on findings by Jim Driscoll
	// http://weblogs.java.net/blog/driscoll/archive/2009/09/08/eval-javascript-global-context
	globalEval: function( data ) {
		if ( data && rnotwhite.test( data ) ) {
			// We use execScript on Internet Explorer
			// We use an anonymous function so that context is window
			// rather than jQuery in Firefox
			( window.execScript || function( data ) {
				window[ "eval" ].call( window, data );
			} )( data );
		}
	},

	// Convert dashed to camelCase; used by the css and data modules
	// Microsoft forgot to hump their vendor prefix (#9572)
	camelCase: function( string ) {
		return string.replace( rmsPrefix, "ms-" ).replace( rdashAlpha, fcamelCase );
	},

	nodeName: function( elem, name ) {
		return elem.nodeName && elem.nodeName.toUpperCase() === name.toUpperCase();
	},

	// args is for internal usage only
	each: function( object, callback, args ) {
		var name, i = 0,
			length = object.length,
			isObj = length === undefined || jQuery.isFunction( object );

		if ( args ) {
			if ( isObj ) {
				for ( name in object ) {
					if ( callback.apply( object[ name ], args ) === false ) {
						break;
					}
				}
			} else {
				for ( ; i < length; ) {
					if ( callback.apply( object[ i++ ], args ) === false ) {
						break;
					}
				}
			}

		// A special, fast, case for the most common use of each
		} else {
			if ( isObj ) {
				for ( name in object ) {
					if ( callback.call( object[ name ], name, object[ name ] ) === false ) {
						break;
					}
				}
			} else {
				for ( ; i < length; ) {
					if ( callback.call( object[ i ], i, object[ i++ ] ) === false ) {
						break;
					}
				}
			}
		}

		return object;
	},

	// Use native String.trim function wherever possible
	trim: trim ?
		function( text ) {
			return text == null ?
				"" :
				trim.call( text );
		} :

		// Otherwise use our own trimming functionality
		function( text ) {
			return text == null ?
				"" :
				text.toString().replace( trimLeft, "" ).replace( trimRight, "" );
		},

	// results is for internal usage only
	makeArray: function( array, results ) {
		var ret = results || [];

		if ( array != null ) {
			// The window, strings (and functions) also have 'length'
			// The extra typeof function check is to prevent crashes
			// in Safari 2 (See: #3039)
			// Tweaked logic slightly to handle Blackberry 4.7 RegExp issues #6930
			var type = jQuery.type( array );

			if ( array.length == null || type === "string" || type === "function" || type === "regexp" || jQuery.isWindow( array ) ) {
				push.call( ret, array );
			} else {
				jQuery.merge( ret, array );
			}
		}

		return ret;
	},

	inArray: function( elem, array ) {
		if ( !array ) {
			return -1;
		}

		if ( indexOf ) {
			return indexOf.call( array, elem );
		}

		for ( var i = 0, length = array.length; i < length; i++ ) {
			if ( array[ i ] === elem ) {
				return i;
			}
		}

		return -1;
	},

	merge: function( first, second ) {
		var i = first.length,
			j = 0;

		if ( typeof second.length === "number" ) {
			for ( var l = second.length; j < l; j++ ) {
				first[ i++ ] = second[ j ];
			}

		} else {
			while ( second[j] !== undefined ) {
				first[ i++ ] = second[ j++ ];
			}
		}

		first.length = i;

		return first;
	},

	grep: function( elems, callback, inv ) {
		var ret = [], retVal;
		inv = !!inv;

		// Go through the array, only saving the items
		// that pass the validator function
		for ( var i = 0, length = elems.length; i < length; i++ ) {
			retVal = !!callback( elems[ i ], i );
			if ( inv !== retVal ) {
				ret.push( elems[ i ] );
			}
		}

		return ret;
	},

	// arg is for internal usage only
	map: function( elems, callback, arg ) {
		var value, key, ret = [],
			i = 0,
			length = elems.length,
			// jquery objects are treated as arrays
			isArray = elems instanceof jQuery || length !== undefined && typeof length === "number" && ( ( length > 0 && elems[ 0 ] && elems[ length -1 ] ) || length === 0 || jQuery.isArray( elems ) ) ;

		// Go through the array, translating each of the items to their
		if ( isArray ) {
			for ( ; i < length; i++ ) {
				value = callback( elems[ i ], i, arg );

				if ( value != null ) {
					ret[ ret.length ] = value;
				}
			}

		// Go through every key on the object,
		} else {
			for ( key in elems ) {
				value = callback( elems[ key ], key, arg );

				if ( value != null ) {
					ret[ ret.length ] = value;
				}
			}
		}

		// Flatten any nested arrays
		return ret.concat.apply( [], ret );
	},

	// A global GUID counter for objects
	guid: 1,

	// Bind a function to a context, optionally partially applying any
	// arguments.
	proxy: function( fn, context ) {
		if ( typeof context === "string" ) {
			var tmp = fn[ context ];
			context = fn;
			fn = tmp;
		}

		// Quick check to determine if target is callable, in the spec
		// this throws a TypeError, but we will just return undefined.
		if ( !jQuery.isFunction( fn ) ) {
			return undefined;
		}

		// Simulated bind
		var args = slice.call( arguments, 2 ),
			proxy = function() {
				return fn.apply( context, args.concat( slice.call( arguments ) ) );
			};

		// Set the guid of unique handler to the same of original handler, so it can be removed
		proxy.guid = fn.guid = fn.guid || proxy.guid || jQuery.guid++;

		return proxy;
	},

	// Mutifunctional method to get and set values to a collection
	// The value/s can optionally be executed if it's a function
	access: function( elems, key, value, exec, fn, pass ) {
		var length = elems.length;

		// Setting many attributes
		if ( typeof key === "object" ) {
			for ( var k in key ) {
				jQuery.access( elems, k, key[k], exec, fn, value );
			}
			return elems;
		}

		// Setting one attribute
		if ( value !== undefined ) {
			// Optionally, function values get executed if exec is true
			exec = !pass && exec && jQuery.isFunction(value);

			for ( var i = 0; i < length; i++ ) {
				fn( elems[i], key, exec ? value.call( elems[i], i, fn( elems[i], key ) ) : value, pass );
			}

			return elems;
		}

		// Getting an attribute
		return length ? fn( elems[0], key ) : undefined;
	},

	now: function() {
		return (new Date()).getTime();
	},

	// Use of jQuery.browser is frowned upon.
	// More details: http://docs.jquery.com/Utilities/jQuery.browser
	uaMatch: function( ua ) {
		ua = ua.toLowerCase();

		var match = rwebkit.exec( ua ) ||
			ropera.exec( ua ) ||
			rmsie.exec( ua ) ||
			ua.indexOf("compatible") < 0 && rmozilla.exec( ua ) ||
			[];

		return { browser: match[1] || "", version: match[2] || "0" };
	},

	sub: function() {
		function jQuerySub( selector, context ) {
			return new jQuerySub.fn.init( selector, context );
		}
		jQuery.extend( true, jQuerySub, this );
		jQuerySub.superclass = this;
		jQuerySub.fn = jQuerySub.prototype = this();
		jQuerySub.fn.constructor = jQuerySub;
		jQuerySub.sub = this.sub;
		jQuerySub.fn.init = function init( selector, context ) {
			if ( context && context instanceof jQuery && !(context instanceof jQuerySub) ) {
				context = jQuerySub( context );
			}

			return jQuery.fn.init.call( this, selector, context, rootjQuerySub );
		};
		jQuerySub.fn.init.prototype = jQuerySub.fn;
		var rootjQuerySub = jQuerySub(document);
		return jQuerySub;
	},

	browser: {}
});

// Populate the class2type map
jQuery.each("Boolean Number String Function Array Date RegExp Object".split(" "), function(i, name) {
	class2type[ "[object " + name + "]" ] = name.toLowerCase();
});

browserMatch = jQuery.uaMatch( userAgent );
if ( browserMatch.browser ) {
	jQuery.browser[ browserMatch.browser ] = true;
	jQuery.browser.version = browserMatch.version;
}

// Deprecated, use jQuery.browser.webkit instead
if ( jQuery.browser.webkit ) {
	jQuery.browser.safari = true;
}

// IE doesn't match non-breaking spaces with \s
if ( rnotwhite.test( "\xA0" ) ) {
	trimLeft = /^[\s\xA0]+/;
	trimRight = /[\s\xA0]+$/;
}

// All jQuery objects should point back to these
rootjQuery = jQuery(document);

// Cleanup functions for the document ready method
if ( document.addEventListener ) {
	DOMContentLoaded = function() {
		document.removeEventListener( "DOMContentLoaded", DOMContentLoaded, false );
		jQuery.ready();
	};

} else if ( document.attachEvent ) {
	DOMContentLoaded = function() {
		// Make sure body exists, at least, in case IE gets a little overzealous (ticket #5443).
		if ( document.readyState === "complete" ) {
			document.detachEvent( "onreadystatechange", DOMContentLoaded );
			jQuery.ready();
		}
	};
}

// The DOM ready check for Internet Explorer
function doScrollCheck() {
	if ( jQuery.isReady ) {
		return;
	}

	try {
		// If IE is used, use the trick by Diego Perini
		// http://javascript.nwbox.com/IEContentLoaded/
		document.documentElement.doScroll("left");
	} catch(e) {
		setTimeout( doScrollCheck, 1 );
		return;
	}

	// and execute any waiting functions
	jQuery.ready();
}

return jQuery;

})();


var // Promise methods
	promiseMethods = "done fail isResolved isRejected promise then always pipe".split( " " ),
	// Static reference to slice
	sliceDeferred = [].slice;

jQuery.extend({
	// Create a simple deferred (one callbacks list)
	_Deferred: function() {
		var // callbacks list
			callbacks = [],
			// stored [ context , args ]
			fired,
			// to avoid firing when already doing so
			firing,
			// flag to know if the deferred has been cancelled
			cancelled,
			// the deferred itself
			deferred  = {

				// done( f1, f2, ...)
				done: function() {
					if ( !cancelled ) {
						var args = arguments,
							i,
							length,
							elem,
							type,
							_fired;
						if ( fired ) {
							_fired = fired;
							fired = 0;
						}
						for ( i = 0, length = args.length; i < length; i++ ) {
							elem = args[ i ];
							type = jQuery.type( elem );
							if ( type === "array" ) {
								deferred.done.apply( deferred, elem );
							} else if ( type === "function" ) {
								callbacks.push( elem );
							}
						}
						if ( _fired ) {
							deferred.resolveWith( _fired[ 0 ], _fired[ 1 ] );
						}
					}
					return this;
				},

				// resolve with given context and args
				resolveWith: function( context, args ) {
					if ( !cancelled && !fired && !firing ) {
						// make sure args are available (#8421)
						args = args || [];
						firing = 1;
						try {
							while( callbacks[ 0 ] ) {
								callbacks.shift().apply( context, args );
							}
						}
						finally {
							fired = [ context, args ];
							firing = 0;
						}
					}
					return this;
				},

				// resolve with this as context and given arguments
				resolve: function() {
					deferred.resolveWith( this, arguments );
					return this;
				},

				// Has this deferred been resolved?
				isResolved: function() {
					return !!( firing || fired );
				},

				// Cancel
				cancel: function() {
					cancelled = 1;
					callbacks = [];
					return this;
				}
			};

		return deferred;
	},

	// Full fledged deferred (two callbacks list)
	Deferred: function( func ) {
		var deferred = jQuery._Deferred(),
			failDeferred = jQuery._Deferred(),
			promise;
		// Add errorDeferred methods, then and promise
		jQuery.extend( deferred, {
			then: function( doneCallbacks, failCallbacks ) {
				deferred.done( doneCallbacks ).fail( failCallbacks );
				return this;
			},
			always: function() {
				return deferred.done.apply( deferred, arguments ).fail.apply( this, arguments );
			},
			fail: failDeferred.done,
			rejectWith: failDeferred.resolveWith,
			reject: failDeferred.resolve,
			isRejected: failDeferred.isResolved,
			pipe: function( fnDone, fnFail ) {
				return jQuery.Deferred(function( newDefer ) {
					jQuery.each( {
						done: [ fnDone, "resolve" ],
						fail: [ fnFail, "reject" ]
					}, function( handler, data ) {
						var fn = data[ 0 ],
							action = data[ 1 ],
							returned;
						if ( jQuery.isFunction( fn ) ) {
							deferred[ handler ](function() {
								returned = fn.apply( this, arguments );
								if ( returned && jQuery.isFunction( returned.promise ) ) {
									returned.promise().then( newDefer.resolve, newDefer.reject );
								} else {
									newDefer[ action + "With" ]( this === deferred ? newDefer : this, [ returned ] );
								}
							});
						} else {
							deferred[ handler ]( newDefer[ action ] );
						}
					});
				}).promise();
			},
			// Get a promise for this deferred
			// If obj is provided, the promise aspect is added to the object
			promise: function( obj ) {
				if ( obj == null ) {
					if ( promise ) {
						return promise;
					}
					promise = obj = {};
				}
				var i = promiseMethods.length;
				while( i-- ) {
					obj[ promiseMethods[i] ] = deferred[ promiseMethods[i] ];
				}
				return obj;
			}
		});
		// Make sure only one callback list will be used
		deferred.done( failDeferred.cancel ).fail( deferred.cancel );
		// Unexpose cancel
		delete deferred.cancel;
		// Call given func if any
		if ( func ) {
			func.call( deferred, deferred );
		}
		return deferred;
	},

	// Deferred helper
	when: function( firstParam ) {
		var args = arguments,
			i = 0,
			length = args.length,
			count = length,
			deferred = length <= 1 && firstParam && jQuery.isFunction( firstParam.promise ) ?
				firstParam :
				jQuery.Deferred();
		function resolveFunc( i ) {
			return function( value ) {
				args[ i ] = arguments.length > 1 ? sliceDeferred.call( arguments, 0 ) : value;
				if ( !( --count ) ) {
					// Strange bug in FF4:
					// Values changed onto the arguments object sometimes end up as undefined values
					// outside the $.when method. Cloning the object into a fresh array solves the issue
					deferred.resolveWith( deferred, sliceDeferred.call( args, 0 ) );
				}
			};
		}
		if ( length > 1 ) {
			for( ; i < length; i++ ) {
				if ( args[ i ] && jQuery.isFunction( args[ i ].promise ) ) {
					args[ i ].promise().then( resolveFunc(i), deferred.reject );
				} else {
					--count;
				}
			}
			if ( !count ) {
				deferred.resolveWith( deferred, args );
			}
		} else if ( deferred !== firstParam ) {
			deferred.resolveWith( deferred, length ? [ firstParam ] : [] );
		}
		return deferred.promise();
	}
});



jQuery.support = (function() {

	var div = document.createElement( "div" ),
		documentElement = document.documentElement,
		all,
		a,
		select,
		opt,
		input,
		marginDiv,
		support,
		fragment,
		body,
		testElementParent,
		testElement,
		testElementStyle,
		tds,
		events,
		eventName,
		i,
		isSupported;

	// Preliminary tests
	div.setAttribute("className", "t");
	div.innerHTML = "   <link><table></table><a href='/a' style='top:1px;float:left;opacity:.55;'>a</a><input type=checkbox>";


	all = div.getElementsByTagName( "*" );
	a = div.getElementsByTagName( "a" )[ 0 ];

	// Can't get basic test support
	if ( !all || !all.length || !a ) {
		return {};
	}

	// First batch of supports tests
	select = document.createElement( "select" );
	opt = select.appendChild( document.createElement("option") );
	input = div.getElementsByTagName( "input" )[ 0 ];

	support = {
		// IE strips leading whitespace when .innerHTML is used
		leadingWhitespace: ( div.firstChild.nodeType === 3 ),

		// Make sure that tbody elements aren't automatically inserted
		// IE will insert them into empty tables
		tbody: !div.getElementsByTagName( "tbody" ).length,

		// Make sure that link elements get serialized correctly by innerHTML
		// This requires a wrapper element in IE
		htmlSerialize: !!div.getElementsByTagName( "link" ).length,

		// Get the style information from getAttribute
		// (IE uses .cssText instead)
		style: /top/.test( a.getAttribute("style") ),

		// Make sure that URLs aren't manipulated
		// (IE normalizes it by default)
		hrefNormalized: ( a.getAttribute( "href" ) === "/a" ),

		// Make sure that element opacity exists
		// (IE uses filter instead)
		// Use a regex to work around a WebKit issue. See #5145
		opacity: /^0.55$/.test( a.style.opacity ),

		// Verify style float existence
		// (IE uses styleFloat instead of cssFloat)
		cssFloat: !!a.style.cssFloat,

		// Make sure that if no value is specified for a checkbox
		// that it defaults to "on".
		// (WebKit defaults to "" instead)
		checkOn: ( input.value === "on" ),

		// Make sure that a selected-by-default option has a working selected property.
		// (WebKit defaults to false instead of true, IE too, if it's in an optgroup)
		optSelected: opt.selected,

		// Test setAttribute on camelCase class. If it works, we need attrFixes when doing get/setAttribute (ie6/7)
		getSetAttribute: div.className !== "t",

		// Will be defined later
		submitBubbles: true,
		changeBubbles: true,
		focusinBubbles: false,
		deleteExpando: true,
		noCloneEvent: true,
		inlineBlockNeedsLayout: false,
		shrinkWrapBlocks: false,
		reliableMarginRight: true
	};

	// Make sure checked status is properly cloned
	input.checked = true;
	support.noCloneChecked = input.cloneNode( true ).checked;

	// Make sure that the options inside disabled selects aren't marked as disabled
	// (WebKit marks them as disabled)
	select.disabled = true;
	support.optDisabled = !opt.disabled;

	// Test to see if it's possible to delete an expando from an element
	// Fails in Internet Explorer
	try {
		delete div.test;
	} catch( e ) {
		support.deleteExpando = false;
	}

	if ( !div.addEventListener && div.attachEvent && div.fireEvent ) {
		div.attachEvent( "onclick", function() {
			// Cloning a node shouldn't copy over any
			// bound event handlers (IE does this)
			support.noCloneEvent = false;
		});
		div.cloneNode( true ).fireEvent( "onclick" );
	}

	// Check if a radio maintains it's value
	// after being appended to the DOM
	input = document.createElement("input");
	input.value = "t";
	input.setAttribute("type", "radio");
	support.radioValue = input.value === "t";

	input.setAttribute("checked", "checked");
	div.appendChild( input );
	fragment = document.createDocumentFragment();
	fragment.appendChild( div.firstChild );

	// WebKit doesn't clone checked state correctly in fragments
	support.checkClone = fragment.cloneNode( true ).cloneNode( true ).lastChild.checked;

	div.innerHTML = "";

	// Figure out if the W3C box model works as expected
	div.style.width = div.style.paddingLeft = "1px";

	body = document.getElementsByTagName( "body" )[ 0 ];
	// We use our own, invisible, body unless the body is already present
	// in which case we use a div (#9239)
	testElement = document.createElement( body ? "div" : "body" );
	testElementStyle = {
		visibility: "hidden",
		width: 0,
		height: 0,
		border: 0,
		margin: 0,
		background: "none"
	};
	if ( body ) {
		jQuery.extend( testElementStyle, {
			position: "absolute",
			left: "-1000px",
			top: "-1000px"
		});
	}
	for ( i in testElementStyle ) {
		testElement.style[ i ] = testElementStyle[ i ];
	}
	testElement.appendChild( div );
	testElementParent = body || documentElement;
	testElementParent.insertBefore( testElement, testElementParent.firstChild );

	// Check if a disconnected checkbox will retain its checked
	// value of true after appended to the DOM (IE6/7)
	support.appendChecked = input.checked;

	support.boxModel = div.offsetWidth === 2;

	if ( "zoom" in div.style ) {
		// Check if natively block-level elements act like inline-block
		// elements when setting their display to 'inline' and giving
		// them layout
		// (IE < 8 does this)
		div.style.display = "inline";
		div.style.zoom = 1;
		support.inlineBlockNeedsLayout = ( div.offsetWidth === 2 );

		// Check if elements with layout shrink-wrap their children
		// (IE 6 does this)
		div.style.display = "";
		div.innerHTML = "<div style='width:4px;'></div>";
		support.shrinkWrapBlocks = ( div.offsetWidth !== 2 );
	}

	div.innerHTML = "<table><tr><td style='padding:0;border:0;display:none'></td><td>t</td></tr></table>";
	tds = div.getElementsByTagName( "td" );

	// Check if table cells still have offsetWidth/Height when they are set
	// to display:none and there are still other visible table cells in a
	// table row; if so, offsetWidth/Height are not reliable for use when
	// determining if an element has been hidden directly using
	// display:none (it is still safe to use offsets if a parent element is
	// hidden; don safety goggles and see bug #4512 for more information).
	// (only IE 8 fails this test)
	isSupported = ( tds[ 0 ].offsetHeight === 0 );

	tds[ 0 ].style.display = "";
	tds[ 1 ].style.display = "none";

	// Check if empty table cells still have offsetWidth/Height
	// (IE < 8 fail this test)
	support.reliableHiddenOffsets = isSupported && ( tds[ 0 ].offsetHeight === 0 );
	div.innerHTML = "";

	// Check if div with explicit width and no margin-right incorrectly
	// gets computed margin-right based on width of container. For more
	// info see bug #3333
	// Fails in WebKit before Feb 2011 nightlies
	// WebKit Bug 13343 - getComputedStyle returns wrong value for margin-right
	if ( document.defaultView && document.defaultView.getComputedStyle ) {
		marginDiv = document.createElement( "div" );
		marginDiv.style.width = "0";
		marginDiv.style.marginRight = "0";
		div.appendChild( marginDiv );
		support.reliableMarginRight =
			( parseInt( ( document.defaultView.getComputedStyle( marginDiv, null ) || { marginRight: 0 } ).marginRight, 10 ) || 0 ) === 0;
	}

	// Remove the body element we added
	testElement.innerHTML = "";
	testElementParent.removeChild( testElement );

	// Technique from Juriy Zaytsev
	// http://thinkweb2.com/projects/prototype/detecting-event-support-without-browser-sniffing/
	// We only care about the case where non-standard event systems
	// are used, namely in IE. Short-circuiting here helps us to
	// avoid an eval call (in setAttribute) which can cause CSP
	// to go haywire. See: https://developer.mozilla.org/en/Security/CSP
	if ( div.attachEvent ) {
		for( i in {
			submit: 1,
			change: 1,
			focusin: 1
		} ) {
			eventName = "on" + i;
			isSupported = ( eventName in div );
			if ( !isSupported ) {
				div.setAttribute( eventName, "return;" );
				isSupported = ( typeof div[ eventName ] === "function" );
			}
			support[ i + "Bubbles" ] = isSupported;
		}
	}

	// Null connected elements to avoid leaks in IE
	testElement = fragment = select = opt = body = marginDiv = div = input = null;

	return support;
})();

// Keep track of boxModel
jQuery.boxModel = jQuery.support.boxModel;




var rbrace = /^(?:\{.*\}|\[.*\])$/,
	rmultiDash = /([a-z])([A-Z])/g;

jQuery.extend({
	cache: {},

	// Please use with caution
	uuid: 0,

	// Unique for each copy of jQuery on the page
	// Non-digits removed to match rinlinejQuery
	expando: "jQuery" + ( jQuery.fn.jquery + Math.random() ).replace( /\D/g, "" ),

	// The following elements throw uncatchable exceptions if you
	// attempt to add expando properties to them.
	noData: {
		"embed": true,
		// Ban all objects except for Flash (which handle expandos)
		"object": "clsid:D27CDB6E-AE6D-11cf-96B8-444553540000",
		"applet": true
	},

	hasData: function( elem ) {
		elem = elem.nodeType ? jQuery.cache[ elem[jQuery.expando] ] : elem[ jQuery.expando ];

		return !!elem && !isEmptyDataObject( elem );
	},

	data: function( elem, name, data, pvt /* Internal Use Only */ ) {
		if ( !jQuery.acceptData( elem ) ) {
			return;
		}

		var thisCache, ret,
			internalKey = jQuery.expando,
			getByName = typeof name === "string",

			// We have to handle DOM nodes and JS objects differently because IE6-7
			// can't GC object references properly across the DOM-JS boundary
			isNode = elem.nodeType,

			// Only DOM nodes need the global jQuery cache; JS object data is
			// attached directly to the object so GC can occur automatically
			cache = isNode ? jQuery.cache : elem,

			// Only defining an ID for JS objects if its cache already exists allows
			// the code to shortcut on the same path as a DOM node with no cache
			id = isNode ? elem[ jQuery.expando ] : elem[ jQuery.expando ] && jQuery.expando;

		// Avoid doing any more work than we need to when trying to get data on an
		// object that has no data at all
		if ( (!id || (pvt && id && (cache[ id ] && !cache[ id ][ internalKey ]))) && getByName && data === undefined ) {
			return;
		}

		if ( !id ) {
			// Only DOM nodes need a new unique ID for each element since their data
			// ends up in the global cache
			if ( isNode ) {
				elem[ jQuery.expando ] = id = ++jQuery.uuid;
			} else {
				id = jQuery.expando;
			}
		}

		if ( !cache[ id ] ) {
			cache[ id ] = {};

			// TODO: This is a hack for 1.5 ONLY. Avoids exposing jQuery
			// metadata on plain JS objects when the object is serialized using
			// JSON.stringify
			if ( !isNode ) {
				cache[ id ].toJSON = jQuery.noop;
			}
		}

		// An object can be passed to jQuery.data instead of a key/value pair; this gets
		// shallow copied over onto the existing cache
		if ( typeof name === "object" || typeof name === "function" ) {
			if ( pvt ) {
				cache[ id ][ internalKey ] = jQuery.extend(cache[ id ][ internalKey ], name);
			} else {
				cache[ id ] = jQuery.extend(cache[ id ], name);
			}
		}

		thisCache = cache[ id ];

		// Internal jQuery data is stored in a separate object inside the object's data
		// cache in order to avoid key collisions between internal data and user-defined
		// data
		if ( pvt ) {
			if ( !thisCache[ internalKey ] ) {
				thisCache[ internalKey ] = {};
			}

			thisCache = thisCache[ internalKey ];
		}

		if ( data !== undefined ) {
			thisCache[ jQuery.camelCase( name ) ] = data;
		}

		// TODO: This is a hack for 1.5 ONLY. It will be removed in 1.6. Users should
		// not attempt to inspect the internal events object using jQuery.data, as this
		// internal data object is undocumented and subject to change.
		if ( name === "events" && !thisCache[name] ) {
			return thisCache[ internalKey ] && thisCache[ internalKey ].events;
		}

		// Check for both converted-to-camel and non-converted data property names
		// If a data property was specified
		if ( getByName ) {

			// First Try to find as-is property data
			ret = thisCache[ name ];

			// Test for null|undefined property data
			if ( ret == null ) {

				// Try to find the camelCased property
				ret = thisCache[ jQuery.camelCase( name ) ];
			}
		} else {
			ret = thisCache;
		}

		return ret;
	},

	removeData: function( elem, name, pvt /* Internal Use Only */ ) {
		if ( !jQuery.acceptData( elem ) ) {
			return;
		}

		var thisCache,

			// Reference to internal data cache key
			internalKey = jQuery.expando,

			isNode = elem.nodeType,

			// See jQuery.data for more information
			cache = isNode ? jQuery.cache : elem,

			// See jQuery.data for more information
			id = isNode ? elem[ jQuery.expando ] : jQuery.expando;

		// If there is already no cache entry for this object, there is no
		// purpose in continuing
		if ( !cache[ id ] ) {
			return;
		}

		if ( name ) {

			thisCache = pvt ? cache[ id ][ internalKey ] : cache[ id ];

			if ( thisCache ) {

				// Support interoperable removal of hyphenated or camelcased keys
				if ( !thisCache[ name ] ) {
					name = jQuery.camelCase( name );
				}

				delete thisCache[ name ];

				// If there is no data left in the cache, we want to continue
				// and let the cache object itself get destroyed
				if ( !isEmptyDataObject(thisCache) ) {
					return;
				}
			}
		}

		// See jQuery.data for more information
		if ( pvt ) {
			delete cache[ id ][ internalKey ];

			// Don't destroy the parent cache unless the internal data object
			// had been the only thing left in it
			if ( !isEmptyDataObject(cache[ id ]) ) {
				return;
			}
		}

		var internalCache = cache[ id ][ internalKey ];

		// Browsers that fail expando deletion also refuse to delete expandos on
		// the window, but it will allow it on all other JS objects; other browsers
		// don't care
		// Ensure that `cache` is not a window object #10080
		if ( jQuery.support.deleteExpando || !cache.setInterval ) {
			delete cache[ id ];
		} else {
			cache[ id ] = null;
		}

		// We destroyed the entire user cache at once because it's faster than
		// iterating through each key, but we need to continue to persist internal
		// data if it existed
		if ( internalCache ) {
			cache[ id ] = {};
			// TODO: This is a hack for 1.5 ONLY. Avoids exposing jQuery
			// metadata on plain JS objects when the object is serialized using
			// JSON.stringify
			if ( !isNode ) {
				cache[ id ].toJSON = jQuery.noop;
			}

			cache[ id ][ internalKey ] = internalCache;

		// Otherwise, we need to eliminate the expando on the node to avoid
		// false lookups in the cache for entries that no longer exist
		} else if ( isNode ) {
			// IE does not allow us to delete expando properties from nodes,
			// nor does it have a removeAttribute function on Document nodes;
			// we must handle all of these cases
			if ( jQuery.support.deleteExpando ) {
				delete elem[ jQuery.expando ];
			} else if ( elem.removeAttribute ) {
				elem.removeAttribute( jQuery.expando );
			} else {
				elem[ jQuery.expando ] = null;
			}
		}
	},

	// For internal use only.
	_data: function( elem, name, data ) {
		return jQuery.data( elem, name, data, true );
	},

	// A method for determining if a DOM node can handle the data expando
	acceptData: function( elem ) {
		if ( elem.nodeName ) {
			var match = jQuery.noData[ elem.nodeName.toLowerCase() ];

			if ( match ) {
				return !(match === true || elem.getAttribute("classid") !== match);
			}
		}

		return true;
	}
});

jQuery.fn.extend({
	data: function( key, value ) {
		var data = null;

		if ( typeof key === "undefined" ) {
			if ( this.length ) {
				data = jQuery.data( this[0] );

				if ( this[0].nodeType === 1 ) {
			    var attr = this[0].attributes, name;
					for ( var i = 0, l = attr.length; i < l; i++ ) {
						name = attr[i].name;

						if ( name.indexOf( "data-" ) === 0 ) {
							name = jQuery.camelCase( name.substring(5) );

							dataAttr( this[0], name, data[ name ] );
						}
					}
				}
			}

			return data;

		} else if ( typeof key === "object" ) {
			return this.each(function() {
				jQuery.data( this, key );
			});
		}

		var parts = key.split(".");
		parts[1] = parts[1] ? "." + parts[1] : "";

		if ( value === undefined ) {
			data = this.triggerHandler("getData" + parts[1] + "!", [parts[0]]);

			// Try to fetch any internally stored data first
			if ( data === undefined && this.length ) {
				data = jQuery.data( this[0], key );
				data = dataAttr( this[0], key, data );
			}

			return data === undefined && parts[1] ?
				this.data( parts[0] ) :
				data;

		} else {
			return this.each(function() {
				var $this = jQuery( this ),
					args = [ parts[0], value ];

				$this.triggerHandler( "setData" + parts[1] + "!", args );
				jQuery.data( this, key, value );
				$this.triggerHandler( "changeData" + parts[1] + "!", args );
			});
		}
	},

	removeData: function( key ) {
		return this.each(function() {
			jQuery.removeData( this, key );
		});
	}
});

function dataAttr( elem, key, data ) {
	// If nothing was found internally, try to fetch any
	// data from the HTML5 data-* attribute
	if ( data === undefined && elem.nodeType === 1 ) {
		var name = "data-" + key.replace( rmultiDash, "$1-$2" ).toLowerCase();

		data = elem.getAttribute( name );

		if ( typeof data === "string" ) {
			try {
				data = data === "true" ? true :
				data === "false" ? false :
				data === "null" ? null :
				!jQuery.isNaN( data ) ? parseFloat( data ) :
					rbrace.test( data ) ? jQuery.parseJSON( data ) :
					data;
			} catch( e ) {}

			// Make sure we set the data so it isn't changed later
			jQuery.data( elem, key, data );

		} else {
			data = undefined;
		}
	}

	return data;
}

// TODO: This is a hack for 1.5 ONLY to allow objects with a single toJSON
// property to be considered empty objects; this property always exists in
// order to make sure JSON.stringify does not expose internal metadata
function isEmptyDataObject( obj ) {
	for ( var name in obj ) {
		if ( name !== "toJSON" ) {
			return false;
		}
	}

	return true;
}




function handleQueueMarkDefer( elem, type, src ) {
	var deferDataKey = type + "defer",
		queueDataKey = type + "queue",
		markDataKey = type + "mark",
		defer = jQuery.data( elem, deferDataKey, undefined, true );
	if ( defer &&
		( src === "queue" || !jQuery.data( elem, queueDataKey, undefined, true ) ) &&
		( src === "mark" || !jQuery.data( elem, markDataKey, undefined, true ) ) ) {
		// Give room for hard-coded callbacks to fire first
		// and eventually mark/queue something else on the element
		setTimeout( function() {
			if ( !jQuery.data( elem, queueDataKey, undefined, true ) &&
				!jQuery.data( elem, markDataKey, undefined, true ) ) {
				jQuery.removeData( elem, deferDataKey, true );
				defer.resolve();
			}
		}, 0 );
	}
}

jQuery.extend({

	_mark: function( elem, type ) {
		if ( elem ) {
			type = (type || "fx") + "mark";
			jQuery.data( elem, type, (jQuery.data(elem,type,undefined,true) || 0) + 1, true );
		}
	},

	_unmark: function( force, elem, type ) {
		if ( force !== true ) {
			type = elem;
			elem = force;
			force = false;
		}
		if ( elem ) {
			type = type || "fx";
			var key = type + "mark",
				count = force ? 0 : ( (jQuery.data( elem, key, undefined, true) || 1 ) - 1 );
			if ( count ) {
				jQuery.data( elem, key, count, true );
			} else {
				jQuery.removeData( elem, key, true );
				handleQueueMarkDefer( elem, type, "mark" );
			}
		}
	},

	queue: function( elem, type, data ) {
		if ( elem ) {
			type = (type || "fx") + "queue";
			var q = jQuery.data( elem, type, undefined, true );
			// Speed up dequeue by getting out quickly if this is just a lookup
			if ( data ) {
				if ( !q || jQuery.isArray(data) ) {
					q = jQuery.data( elem, type, jQuery.makeArray(data), true );
				} else {
					q.push( data );
				}
			}
			return q || [];
		}
	},

	dequeue: function( elem, type ) {
		type = type || "fx";

		var queue = jQuery.queue( elem, type ),
			fn = queue.shift(),
			defer;

		// If the fx queue is dequeued, always remove the progress sentinel
		if ( fn === "inprogress" ) {
			fn = queue.shift();
		}

		if ( fn ) {
			// Add a progress sentinel to prevent the fx queue from being
			// automatically dequeued
			if ( type === "fx" ) {
				queue.unshift("inprogress");
			}

			fn.call(elem, function() {
				jQuery.dequeue(elem, type);
			});
		}

		if ( !queue.length ) {
			jQuery.removeData( elem, type + "queue", true );
			handleQueueMarkDefer( elem, type, "queue" );
		}
	}
});

jQuery.fn.extend({
	queue: function( type, data ) {
		if ( typeof type !== "string" ) {
			data = type;
			type = "fx";
		}

		if ( data === undefined ) {
			return jQuery.queue( this[0], type );
		}
		return this.each(function() {
			var queue = jQuery.queue( this, type, data );

			if ( type === "fx" && queue[0] !== "inprogress" ) {
				jQuery.dequeue( this, type );
			}
		});
	},
	dequeue: function( type ) {
		return this.each(function() {
			jQuery.dequeue( this, type );
		});
	},
	// Based off of the plugin by Clint Helfers, with permission.
	// http://blindsignals.com/index.php/2009/07/jquery-delay/
	delay: function( time, type ) {
		time = jQuery.fx ? jQuery.fx.speeds[time] || time : time;
		type = type || "fx";

		return this.queue( type, function() {
			var elem = this;
			setTimeout(function() {
				jQuery.dequeue( elem, type );
			}, time );
		});
	},
	clearQueue: function( type ) {
		return this.queue( type || "fx", [] );
	},
	// Get a promise resolved when queues of a certain type
	// are emptied (fx is the type by default)
	promise: function( type, object ) {
		if ( typeof type !== "string" ) {
			object = type;
			type = undefined;
		}
		type = type || "fx";
		var defer = jQuery.Deferred(),
			elements = this,
			i = elements.length,
			count = 1,
			deferDataKey = type + "defer",
			queueDataKey = type + "queue",
			markDataKey = type + "mark",
			tmp;
		function resolve() {
			if ( !( --count ) ) {
				defer.resolveWith( elements, [ elements ] );
			}
		}
		while( i-- ) {
			if (( tmp = jQuery.data( elements[ i ], deferDataKey, undefined, true ) ||
					( jQuery.data( elements[ i ], queueDataKey, undefined, true ) ||
						jQuery.data( elements[ i ], markDataKey, undefined, true ) ) &&
					jQuery.data( elements[ i ], deferDataKey, jQuery._Deferred(), true ) )) {
				count++;
				tmp.done( resolve );
			}
		}
		resolve();
		return defer.promise();
	}
});




var rclass = /[\n\t\r]/g,
	rspace = /\s+/,
	rreturn = /\r/g,
	rtype = /^(?:button|input)$/i,
	rfocusable = /^(?:button|input|object|select|textarea)$/i,
	rclickable = /^a(?:rea)?$/i,
	rboolean = /^(?:autofocus|autoplay|async|checked|controls|defer|disabled|hidden|loop|multiple|open|readonly|required|scoped|selected)$/i,
	nodeHook, boolHook;

jQuery.fn.extend({
	attr: function( name, value ) {
		return jQuery.access( this, name, value, true, jQuery.attr );
	},

	removeAttr: function( name ) {
		return this.each(function() {
			jQuery.removeAttr( this, name );
		});
	},
	
	prop: function( name, value ) {
		return jQuery.access( this, name, value, true, jQuery.prop );
	},
	
	removeProp: function( name ) {
		name = jQuery.propFix[ name ] || name;
		return this.each(function() {
			// try/catch handles cases where IE balks (such as removing a property on window)
			try {
				this[ name ] = undefined;
				delete this[ name ];
			} catch( e ) {}
		});
	},

	addClass: function( value ) {
		var classNames, i, l, elem,
			setClass, c, cl;

		if ( jQuery.isFunction( value ) ) {
			return this.each(function( j ) {
				jQuery( this ).addClass( value.call(this, j, this.className) );
			});
		}

		if ( value && typeof value === "string" ) {
			classNames = value.split( rspace );

			for ( i = 0, l = this.length; i < l; i++ ) {
				elem = this[ i ];

				if ( elem.nodeType === 1 ) {
					if ( !elem.className && classNames.length === 1 ) {
						elem.className = value;

					} else {
						setClass = " " + elem.className + " ";

						for ( c = 0, cl = classNames.length; c < cl; c++ ) {
							if ( !~setClass.indexOf( " " + classNames[ c ] + " " ) ) {
								setClass += classNames[ c ] + " ";
							}
						}
						elem.className = jQuery.trim( setClass );
					}
				}
			}
		}

		return this;
	},

	removeClass: function( value ) {
		var classNames, i, l, elem, className, c, cl;

		if ( jQuery.isFunction( value ) ) {
			return this.each(function( j ) {
				jQuery( this ).removeClass( value.call(this, j, this.className) );
			});
		}

		if ( (value && typeof value === "string") || value === undefined ) {
			classNames = (value || "").split( rspace );

			for ( i = 0, l = this.length; i < l; i++ ) {
				elem = this[ i ];

				if ( elem.nodeType === 1 && elem.className ) {
					if ( value ) {
						className = (" " + elem.className + " ").replace( rclass, " " );
						for ( c = 0, cl = classNames.length; c < cl; c++ ) {
							className = className.replace(" " + classNames[ c ] + " ", " ");
						}
						elem.className = jQuery.trim( className );

					} else {
						elem.className = "";
					}
				}
			}
		}

		return this;
	},

	toggleClass: function( value, stateVal ) {
		var type = typeof value,
			isBool = typeof stateVal === "boolean";

		if ( jQuery.isFunction( value ) ) {
			return this.each(function( i ) {
				jQuery( this ).toggleClass( value.call(this, i, this.className, stateVal), stateVal );
			});
		}

		return this.each(function() {
			if ( type === "string" ) {
				// toggle individual class names
				var className,
					i = 0,
					self = jQuery( this ),
					state = stateVal,
					classNames = value.split( rspace );

				while ( (className = classNames[ i++ ]) ) {
					// check each className given, space seperated list
					state = isBool ? state : !self.hasClass( className );
					self[ state ? "addClass" : "removeClass" ]( className );
				}

			} else if ( type === "undefined" || type === "boolean" ) {
				if ( this.className ) {
					// store className if set
					jQuery._data( this, "__className__", this.className );
				}

				// toggle whole className
				this.className = this.className || value === false ? "" : jQuery._data( this, "__className__" ) || "";
			}
		});
	},

	hasClass: function( selector ) {
		var className = " " + selector + " ";
		for ( var i = 0, l = this.length; i < l; i++ ) {
			if ( this[i].nodeType === 1 && (" " + this[i].className + " ").replace(rclass, " ").indexOf( className ) > -1 ) {
				return true;
			}
		}

		return false;
	},

	val: function( value ) {
		var hooks, ret,
			elem = this[0];
		
		if ( !arguments.length ) {
			if ( elem ) {
				hooks = jQuery.valHooks[ elem.nodeName.toLowerCase() ] || jQuery.valHooks[ elem.type ];

				if ( hooks && "get" in hooks && (ret = hooks.get( elem, "value" )) !== undefined ) {
					return ret;
				}

				ret = elem.value;

				return typeof ret === "string" ? 
					// handle most common string cases
					ret.replace(rreturn, "") : 
					// handle cases where value is null/undef or number
					ret == null ? "" : ret;
			}

			return undefined;
		}

		var isFunction = jQuery.isFunction( value );

		return this.each(function( i ) {
			var self = jQuery(this), val;

			if ( this.nodeType !== 1 ) {
				return;
			}

			if ( isFunction ) {
				val = value.call( this, i, self.val() );
			} else {
				val = value;
			}

			// Treat null/undefined as ""; convert numbers to string
			if ( val == null ) {
				val = "";
			} else if ( typeof val === "number" ) {
				val += "";
			} else if ( jQuery.isArray( val ) ) {
				val = jQuery.map(val, function ( value ) {
					return value == null ? "" : value + "";
				});
			}

			hooks = jQuery.valHooks[ this.nodeName.toLowerCase() ] || jQuery.valHooks[ this.type ];

			// If set returns undefined, fall back to normal setting
			if ( !hooks || !("set" in hooks) || hooks.set( this, val, "value" ) === undefined ) {
				this.value = val;
			}
		});
	}
});

jQuery.extend({
	valHooks: {
		option: {
			get: function( elem ) {
				// attributes.value is undefined in Blackberry 4.7 but
				// uses .value. See #6932
				var val = elem.attributes.value;
				return !val || val.specified ? elem.value : elem.text;
			}
		},
		select: {
			get: function( elem ) {
				var value,
					index = elem.selectedIndex,
					values = [],
					options = elem.options,
					one = elem.type === "select-one";

				// Nothing was selected
				if ( index < 0 ) {
					return null;
				}

				// Loop through all the selected options
				for ( var i = one ? index : 0, max = one ? index + 1 : options.length; i < max; i++ ) {
					var option = options[ i ];

					// Don't return options that are disabled or in a disabled optgroup
					if ( option.selected && (jQuery.support.optDisabled ? !option.disabled : option.getAttribute("disabled") === null) &&
							(!option.parentNode.disabled || !jQuery.nodeName( option.parentNode, "optgroup" )) ) {

						// Get the specific value for the option
						value = jQuery( option ).val();

						// We don't need an array for one selects
						if ( one ) {
							return value;
						}

						// Multi-Selects return an array
						values.push( value );
					}
				}

				// Fixes Bug #2551 -- select.val() broken in IE after form.reset()
				if ( one && !values.length && options.length ) {
					return jQuery( options[ index ] ).val();
				}

				return values;
			},

			set: function( elem, value ) {
				var values = jQuery.makeArray( value );

				jQuery(elem).find("option").each(function() {
					this.selected = jQuery.inArray( jQuery(this).val(), values ) >= 0;
				});

				if ( !values.length ) {
					elem.selectedIndex = -1;
				}
				return values;
			}
		}
	},

	attrFn: {
		val: true,
		css: true,
		html: true,
		text: true,
		data: true,
		width: true,
		height: true,
		offset: true
	},
	
	attrFix: {
		// Always normalize to ensure hook usage
		tabindex: "tabIndex"
	},
	
	attr: function( elem, name, value, pass ) {
		var nType = elem.nodeType;
		
		// don't get/set attributes on text, comment and attribute nodes
		if ( !elem || nType === 3 || nType === 8 || nType === 2 ) {
			return undefined;
		}

		if ( pass && name in jQuery.attrFn ) {
			return jQuery( elem )[ name ]( value );
		}

		// Fallback to prop when attributes are not supported
		if ( !("getAttribute" in elem) ) {
			return jQuery.prop( elem, name, value );
		}

		var ret, hooks,
			notxml = nType !== 1 || !jQuery.isXMLDoc( elem );

		// Normalize the name if needed
		if ( notxml ) {
			name = jQuery.attrFix[ name ] || name;

			hooks = jQuery.attrHooks[ name ];

			if ( !hooks ) {
				// Use boolHook for boolean attributes
				if ( rboolean.test( name ) ) {
					hooks = boolHook;

				// Use nodeHook if available( IE6/7 )
				} else if ( nodeHook ) {
					hooks = nodeHook;
				}
			}
		}

		if ( value !== undefined ) {

			if ( value === null ) {
				jQuery.removeAttr( elem, name );
				return undefined;

			} else if ( hooks && "set" in hooks && notxml && (ret = hooks.set( elem, value, name )) !== undefined ) {
				return ret;

			} else {
				elem.setAttribute( name, "" + value );
				return value;
			}

		} else if ( hooks && "get" in hooks && notxml && (ret = hooks.get( elem, name )) !== null ) {
			return ret;

		} else {

			ret = elem.getAttribute( name );

			// Non-existent attributes return null, we normalize to undefined
			return ret === null ?
				undefined :
				ret;
		}
	},

	removeAttr: function( elem, name ) {
		var propName;
		if ( elem.nodeType === 1 ) {
			name = jQuery.attrFix[ name ] || name;

			jQuery.attr( elem, name, "" );
			elem.removeAttribute( name );

			// Set corresponding property to false for boolean attributes
			if ( rboolean.test( name ) && (propName = jQuery.propFix[ name ] || name) in elem ) {
				elem[ propName ] = false;
			}
		}
	},

	attrHooks: {
		type: {
			set: function( elem, value ) {
				// We can't allow the type property to be changed (since it causes problems in IE)
				if ( rtype.test( elem.nodeName ) && elem.parentNode ) {
					jQuery.error( "type property can't be changed" );
				} else if ( !jQuery.support.radioValue && value === "radio" && jQuery.nodeName(elem, "input") ) {
					// Setting the type on a radio button after the value resets the value in IE6-9
					// Reset value to it's default in case type is set after value
					// This is for element creation
					var val = elem.value;
					elem.setAttribute( "type", value );
					if ( val ) {
						elem.value = val;
					}
					return value;
				}
			}
		},
		// Use the value property for back compat
		// Use the nodeHook for button elements in IE6/7 (#1954)
		value: {
			get: function( elem, name ) {
				if ( nodeHook && jQuery.nodeName( elem, "button" ) ) {
					return nodeHook.get( elem, name );
				}
				return name in elem ?
					elem.value :
					null;
			},
			set: function( elem, value, name ) {
				if ( nodeHook && jQuery.nodeName( elem, "button" ) ) {
					return nodeHook.set( elem, value, name );
				}
				// Does not return so that setAttribute is also used
				elem.value = value;
			}
		}
	},

	propFix: {
		tabindex: "tabIndex",
		readonly: "readOnly",
		"for": "htmlFor",
		"class": "className",
		maxlength: "maxLength",
		cellspacing: "cellSpacing",
		cellpadding: "cellPadding",
		rowspan: "rowSpan",
		colspan: "colSpan",
		usemap: "useMap",
		frameborder: "frameBorder",
		contenteditable: "contentEditable"
	},
	
	prop: function( elem, name, value ) {
		var nType = elem.nodeType;

		// don't get/set properties on text, comment and attribute nodes
		if ( !elem || nType === 3 || nType === 8 || nType === 2 ) {
			return undefined;
		}

		var ret, hooks,
			notxml = nType !== 1 || !jQuery.isXMLDoc( elem );

		if ( notxml ) {
			// Fix name and attach hooks
			name = jQuery.propFix[ name ] || name;
			hooks = jQuery.propHooks[ name ];
		}

		if ( value !== undefined ) {
			if ( hooks && "set" in hooks && (ret = hooks.set( elem, value, name )) !== undefined ) {
				return ret;

			} else {
				return (elem[ name ] = value);
			}

		} else {
			if ( hooks && "get" in hooks && (ret = hooks.get( elem, name )) !== null ) {
				return ret;

			} else {
				return elem[ name ];
			}
		}
	},
	
	propHooks: {
		tabIndex: {
			get: function( elem ) {
				// elem.tabIndex doesn't always return the correct value when it hasn't been explicitly set
				// http://fluidproject.org/blog/2008/01/09/getting-setting-and-removing-tabindex-values-with-javascript/
				var attributeNode = elem.getAttributeNode("tabindex");

				return attributeNode && attributeNode.specified ?
					parseInt( attributeNode.value, 10 ) :
					rfocusable.test( elem.nodeName ) || rclickable.test( elem.nodeName ) && elem.href ?
						0 :
						undefined;
			}
		}
	}
});

// Add the tabindex propHook to attrHooks for back-compat
jQuery.attrHooks.tabIndex = jQuery.propHooks.tabIndex;

// Hook for boolean attributes
boolHook = {
	get: function( elem, name ) {
		// Align boolean attributes with corresponding properties
		// Fall back to attribute presence where some booleans are not supported
		var attrNode;
		return jQuery.prop( elem, name ) === true || ( attrNode = elem.getAttributeNode( name ) ) && attrNode.nodeValue !== false ?
			name.toLowerCase() :
			undefined;
	},
	set: function( elem, value, name ) {
		var propName;
		if ( value === false ) {
			// Remove boolean attributes when set to false
			jQuery.removeAttr( elem, name );
		} else {
			// value is true since we know at this point it's type boolean and not false
			// Set boolean attributes to the same name and set the DOM property
			propName = jQuery.propFix[ name ] || name;
			if ( propName in elem ) {
				// Only set the IDL specifically if it already exists on the element
				elem[ propName ] = true;
			}

			elem.setAttribute( name, name.toLowerCase() );
		}
		return name;
	}
};

// IE6/7 do not support getting/setting some attributes with get/setAttribute
if ( !jQuery.support.getSetAttribute ) {
	
	// Use this for any attribute in IE6/7
	// This fixes almost every IE6/7 issue
	nodeHook = jQuery.valHooks.button = {
		get: function( elem, name ) {
			var ret;
			ret = elem.getAttributeNode( name );
			// Return undefined if nodeValue is empty string
			return ret && ret.nodeValue !== "" ?
				ret.nodeValue :
				undefined;
		},
		set: function( elem, value, name ) {
			// Set the existing or create a new attribute node
			var ret = elem.getAttributeNode( name );
			if ( !ret ) {
				ret = document.createAttribute( name );
				elem.setAttributeNode( ret );
			}
			return (ret.nodeValue = value + "");
		}
	};

	// Set width and height to auto instead of 0 on empty string( Bug #8150 )
	// This is for removals
	jQuery.each([ "width", "height" ], function( i, name ) {
		jQuery.attrHooks[ name ] = jQuery.extend( jQuery.attrHooks[ name ], {
			set: function( elem, value ) {
				if ( value === "" ) {
					elem.setAttribute( name, "auto" );
					return value;
				}
			}
		});
	});
}


// Some attributes require a special call on IE
if ( !jQuery.support.hrefNormalized ) {
	jQuery.each([ "href", "src", "width", "height" ], function( i, name ) {
		jQuery.attrHooks[ name ] = jQuery.extend( jQuery.attrHooks[ name ], {
			get: function( elem ) {
				var ret = elem.getAttribute( name, 2 );
				return ret === null ? undefined : ret;
			}
		});
	});
}

if ( !jQuery.support.style ) {
	jQuery.attrHooks.style = {
		get: function( elem ) {
			// Return undefined in the case of empty string
			// Normalize to lowercase since IE uppercases css property names
			return elem.style.cssText.toLowerCase() || undefined;
		},
		set: function( elem, value ) {
			return (elem.style.cssText = "" + value);
		}
	};
}

// Safari mis-reports the default selected property of an option
// Accessing the parent's selectedIndex property fixes it
if ( !jQuery.support.optSelected ) {
	jQuery.propHooks.selected = jQuery.extend( jQuery.propHooks.selected, {
		get: function( elem ) {
			var parent = elem.parentNode;

			if ( parent ) {
				parent.selectedIndex;

				// Make sure that it also works with optgroups, see #5701
				if ( parent.parentNode ) {
					parent.parentNode.selectedIndex;
				}
			}
			return null;
		}
	});
}

// Radios and checkboxes getter/setter
if ( !jQuery.support.checkOn ) {
	jQuery.each([ "radio", "checkbox" ], function() {
		jQuery.valHooks[ this ] = {
			get: function( elem ) {
				// Handle the case where in Webkit "" is returned instead of "on" if a value isn't specified
				return elem.getAttribute("value") === null ? "on" : elem.value;
			}
		};
	});
}
jQuery.each([ "radio", "checkbox" ], function() {
	jQuery.valHooks[ this ] = jQuery.extend( jQuery.valHooks[ this ], {
		set: function( elem, value ) {
			if ( jQuery.isArray( value ) ) {
				return (elem.checked = jQuery.inArray( jQuery(elem).val(), value ) >= 0);
			}
		}
	});
});




var rnamespaces = /\.(.*)$/,
	rformElems = /^(?:textarea|input|select)$/i,
	rperiod = /\./g,
	rspaces = / /g,
	rescape = /[^\w\s.|`]/g,
	fcleanup = function( nm ) {
		return nm.replace(rescape, "\\$&");
	};

/*
 * A number of helper functions used for managing events.
 * Many of the ideas behind this code originated from
 * Dean Edwards' addEvent library.
 */
jQuery.event = {

	// Bind an event to an element
	// Original by Dean Edwards
	add: function( elem, types, handler, data ) {
		if ( elem.nodeType === 3 || elem.nodeType === 8 ) {
			return;
		}

		if ( handler === false ) {
			handler = returnFalse;
		} else if ( !handler ) {
			// Fixes bug #7229. Fix recommended by jdalton
			return;
		}

		var handleObjIn, handleObj;

		if ( handler.handler ) {
			handleObjIn = handler;
			handler = handleObjIn.handler;
		}

		// Make sure that the function being executed has a unique ID
		if ( !handler.guid ) {
			handler.guid = jQuery.guid++;
		}

		// Init the element's event structure
		var elemData = jQuery._data( elem );

		// If no elemData is found then we must be trying to bind to one of the
		// banned noData elements
		if ( !elemData ) {
			return;
		}

		var events = elemData.events,
			eventHandle = elemData.handle;

		if ( !events ) {
			elemData.events = events = {};
		}

		if ( !eventHandle ) {
			elemData.handle = eventHandle = function( e ) {
				// Discard the second event of a jQuery.event.trigger() and
				// when an event is called after a page has unloaded
				return typeof jQuery !== "undefined" && (!e || jQuery.event.triggered !== e.type) ?
					jQuery.event.handle.apply( eventHandle.elem, arguments ) :
					undefined;
			};
		}

		// Add elem as a property of the handle function
		// This is to prevent a memory leak with non-native events in IE.
		eventHandle.elem = elem;

		// Handle multiple events separated by a space
		// jQuery(...).bind("mouseover mouseout", fn);
		types = types.split(" ");

		var type, i = 0, namespaces;

		while ( (type = types[ i++ ]) ) {
			handleObj = handleObjIn ?
				jQuery.extend({}, handleObjIn) :
				{ handler: handler, data: data };

			// Namespaced event handlers
			if ( type.indexOf(".") > -1 ) {
				namespaces = type.split(".");
				type = namespaces.shift();
				handleObj.namespace = namespaces.slice(0).sort().join(".");

			} else {
				namespaces = [];
				handleObj.namespace = "";
			}

			handleObj.type = type;
			if ( !handleObj.guid ) {
				handleObj.guid = handler.guid;
			}

			// Get the current list of functions bound to this event
			var handlers = events[ type ],
				special = jQuery.event.special[ type ] || {};

			// Init the event handler queue
			if ( !handlers ) {
				handlers = events[ type ] = [];

				// Check for a special event handler
				// Only use addEventListener/attachEvent if the special
				// events handler returns false
				if ( !special.setup || special.setup.call( elem, data, namespaces, eventHandle ) === false ) {
					// Bind the global event handler to the element
					if ( elem.addEventListener ) {
						elem.addEventListener( type, eventHandle, false );

					} else if ( elem.attachEvent ) {
						elem.attachEvent( "on" + type, eventHandle );
					}
				}
			}

			if ( special.add ) {
				special.add.call( elem, handleObj );

				if ( !handleObj.handler.guid ) {
					handleObj.handler.guid = handler.guid;
				}
			}

			// Add the function to the element's handler list
			handlers.push( handleObj );

			// Keep track of which events have been used, for event optimization
			jQuery.event.global[ type ] = true;
		}

		// Nullify elem to prevent memory leaks in IE
		elem = null;
	},

	global: {},

	// Detach an event or set of events from an element
	remove: function( elem, types, handler, pos ) {
		// don't do events on text and comment nodes
		if ( elem.nodeType === 3 || elem.nodeType === 8 ) {
			return;
		}

		if ( handler === false ) {
			handler = returnFalse;
		}

		var ret, type, fn, j, i = 0, all, namespaces, namespace, special, eventType, handleObj, origType,
			elemData = jQuery.hasData( elem ) && jQuery._data( elem ),
			events = elemData && elemData.events;

		if ( !elemData || !events ) {
			return;
		}

		// types is actually an event object here
		if ( types && types.type ) {
			handler = types.handler;
			types = types.type;
		}

		// Unbind all events for the element
		if ( !types || typeof types === "string" && types.charAt(0) === "." ) {
			types = types || "";

			for ( type in events ) {
				jQuery.event.remove( elem, type + types );
			}

			return;
		}

		// Handle multiple events separated by a space
		// jQuery(...).unbind("mouseover mouseout", fn);
		types = types.split(" ");

		while ( (type = types[ i++ ]) ) {
			origType = type;
			handleObj = null;
			all = type.indexOf(".") < 0;
			namespaces = [];

			if ( !all ) {
				// Namespaced event handlers
				namespaces = type.split(".");
				type = namespaces.shift();

				namespace = new RegExp("(^|\\.)" +
					jQuery.map( namespaces.slice(0).sort(), fcleanup ).join("\\.(?:.*\\.)?") + "(\\.|$)");
			}

			eventType = events[ type ];

			if ( !eventType ) {
				continue;
			}

			if ( !handler ) {
				for ( j = 0; j < eventType.length; j++ ) {
					handleObj = eventType[ j ];

					if ( all || namespace.test( handleObj.namespace ) ) {
						jQuery.event.remove( elem, origType, handleObj.handler, j );
						eventType.splice( j--, 1 );
					}
				}

				continue;
			}

			special = jQuery.event.special[ type ] || {};

			for ( j = pos || 0; j < eventType.length; j++ ) {
				handleObj = eventType[ j ];

				if ( handler.guid === handleObj.guid ) {
					// remove the given handler for the given type
					if ( all || namespace.test( handleObj.namespace ) ) {
						if ( pos == null ) {
							eventType.splice( j--, 1 );
						}

						if ( special.remove ) {
							special.remove.call( elem, handleObj );
						}
					}

					if ( pos != null ) {
						break;
					}
				}
			}

			// remove generic event handler if no more handlers exist
			if ( eventType.length === 0 || pos != null && eventType.length === 1 ) {
				if ( !special.teardown || special.teardown.call( elem, namespaces ) === false ) {
					jQuery.removeEvent( elem, type, elemData.handle );
				}

				ret = null;
				delete events[ type ];
			}
		}

		// Remove the expando if it's no longer used
		if ( jQuery.isEmptyObject( events ) ) {
			var handle = elemData.handle;
			if ( handle ) {
				handle.elem = null;
			}

			delete elemData.events;
			delete elemData.handle;

			if ( jQuery.isEmptyObject( elemData ) ) {
				jQuery.removeData( elem, undefined, true );
			}
		}
	},
	
	// Events that are safe to short-circuit if no handlers are attached.
	// Native DOM events should not be added, they may have inline handlers.
	customEvent: {
		"getData": true,
		"setData": true,
		"changeData": true
	},

	trigger: function( event, data, elem, onlyHandlers ) {
		// Event object or event type
		var type = event.type || event,
			namespaces = [],
			exclusive;

		if ( type.indexOf("!") >= 0 ) {
			// Exclusive events trigger only for the exact event (no namespaces)
			type = type.slice(0, -1);
			exclusive = true;
		}

		if ( type.indexOf(".") >= 0 ) {
			// Namespaced trigger; create a regexp to match event type in handle()
			namespaces = type.split(".");
			type = namespaces.shift();
			namespaces.sort();
		}

		if ( (!elem || jQuery.event.customEvent[ type ]) && !jQuery.event.global[ type ] ) {
			// No jQuery handlers for this event type, and it can't have inline handlers
			return;
		}

		// Caller can pass in an Event, Object, or just an event type string
		event = typeof event === "object" ?
			// jQuery.Event object
			event[ jQuery.expando ] ? event :
			// Object literal
			new jQuery.Event( type, event ) :
			// Just the event type (string)
			new jQuery.Event( type );

		event.type = type;
		event.exclusive = exclusive;
		event.namespace = namespaces.join(".");
		event.namespace_re = new RegExp("(^|\\.)" + namespaces.join("\\.(?:.*\\.)?") + "(\\.|$)");
		
		// triggerHandler() and global events don't bubble or run the default action
		if ( onlyHandlers || !elem ) {
			event.preventDefault();
			event.stopPropagation();
		}

		// Handle a global trigger
		if ( !elem ) {
			// TODO: Stop taunting the data cache; remove global events and always attach to document
			jQuery.each( jQuery.cache, function() {
				// internalKey variable is just used to make it easier to find
				// and potentially change this stuff later; currently it just
				// points to jQuery.expando
				var internalKey = jQuery.expando,
					internalCache = this[ internalKey ];
				if ( internalCache && internalCache.events && internalCache.events[ type ] ) {
					jQuery.event.trigger( event, data, internalCache.handle.elem );
				}
			});
			return;
		}

		// Don't do events on text and comment nodes
		if ( elem.nodeType === 3 || elem.nodeType === 8 ) {
			return;
		}

		// Clean up the event in case it is being reused
		event.result = undefined;
		event.target = elem;

		// Clone any incoming data and prepend the event, creating the handler arg list
		data = data != null ? jQuery.makeArray( data ) : [];
		data.unshift( event );

		var cur = elem,
			// IE doesn't like method names with a colon (#3533, #8272)
			ontype = type.indexOf(":") < 0 ? "on" + type : "";

		// Fire event on the current element, then bubble up the DOM tree
		do {
			var handle = jQuery._data( cur, "handle" );

			event.currentTarget = cur;
			if ( handle ) {
				handle.apply( cur, data );
			}

			// Trigger an inline bound script
			if ( ontype && jQuery.acceptData( cur ) && cur[ ontype ] && cur[ ontype ].apply( cur, data ) === false ) {
				event.result = false;
				event.preventDefault();
			}

			// Bubble up to document, then to window
			cur = cur.parentNode || cur.ownerDocument || cur === event.target.ownerDocument && window;
		} while ( cur && !event.isPropagationStopped() );

		// If nobody prevented the default action, do it now
		if ( !event.isDefaultPrevented() ) {
			var old,
				special = jQuery.event.special[ type ] || {};

			if ( (!special._default || special._default.call( elem.ownerDocument, event ) === false) &&
				!(type === "click" && jQuery.nodeName( elem, "a" )) && jQuery.acceptData( elem ) ) {

				// Call a native DOM method on the target with the same name name as the event.
				// Can't use an .isFunction)() check here because IE6/7 fails that test.
				// IE<9 dies on focus to hidden element (#1486), may want to revisit a try/catch.
				try {
					if ( ontype && elem[ type ] ) {
						// Don't re-trigger an onFOO event when we call its FOO() method
						old = elem[ ontype ];

						if ( old ) {
							elem[ ontype ] = null;
						}

						jQuery.event.triggered = type;
						elem[ type ]();
					}
				} catch ( ieError ) {}

				if ( old ) {
					elem[ ontype ] = old;
				}

				jQuery.event.triggered = undefined;
			}
		}
		
		return event.result;
	},

	handle: function( event ) {
		event = jQuery.event.fix( event || window.event );
		// Snapshot the handlers list since a called handler may add/remove events.
		var handlers = ((jQuery._data( this, "events" ) || {})[ event.type ] || []).slice(0),
			run_all = !event.exclusive && !event.namespace,
			args = Array.prototype.slice.call( arguments, 0 );

		// Use the fix-ed Event rather than the (read-only) native event
		args[0] = event;
		event.currentTarget = this;

		for ( var j = 0, l = handlers.length; j < l; j++ ) {
			var handleObj = handlers[ j ];

			// Triggered event must 1) be non-exclusive and have no namespace, or
			// 2) have namespace(s) a subset or equal to those in the bound event.
			if ( run_all || event.namespace_re.test( handleObj.namespace ) ) {
				// Pass in a reference to the handler function itself
				// So that we can later remove it
				event.handler = handleObj.handler;
				event.data = handleObj.data;
				event.handleObj = handleObj;

				var ret = handleObj.handler.apply( this, args );

				if ( ret !== undefined ) {
					event.result = ret;
					if ( ret === false ) {
						event.preventDefault();
						event.stopPropagation();
					}
				}

				if ( event.isImmediatePropagationStopped() ) {
					break;
				}
			}
		}
		return event.result;
	},

	props: "altKey attrChange attrName bubbles button cancelable charCode clientX clientY ctrlKey currentTarget data detail eventPhase fromElement handler keyCode layerX layerY metaKey newValue offsetX offsetY pageX pageY prevValue relatedNode relatedTarget screenX screenY shiftKey srcElement target toElement view wheelDelta which".split(" "),

	fix: function( event ) {
		if ( event[ jQuery.expando ] ) {
			return event;
		}

		// store a copy of the original event object
		// and "clone" to set read-only properties
		var originalEvent = event;
		event = jQuery.Event( originalEvent );

		for ( var i = this.props.length, prop; i; ) {
			prop = this.props[ --i ];
			event[ prop ] = originalEvent[ prop ];
		}

		// Fix target property, if necessary
		if ( !event.target ) {
			// Fixes #1925 where srcElement might not be defined either
			event.target = event.srcElement || document;
		}

		// check if target is a textnode (safari)
		if ( event.target.nodeType === 3 ) {
			event.target = event.target.parentNode;
		}

		// Add relatedTarget, if necessary
		if ( !event.relatedTarget && event.fromElement ) {
			event.relatedTarget = event.fromElement === event.target ? event.toElement : event.fromElement;
		}

		// Calculate pageX/Y if missing and clientX/Y available
		if ( event.pageX == null && event.clientX != null ) {
			var eventDocument = event.target.ownerDocument || document,
				doc = eventDocument.documentElement,
				body = eventDocument.body;

			event.pageX = event.clientX + (doc && doc.scrollLeft || body && body.scrollLeft || 0) - (doc && doc.clientLeft || body && body.clientLeft || 0);
			event.pageY = event.clientY + (doc && doc.scrollTop  || body && body.scrollTop  || 0) - (doc && doc.clientTop  || body && body.clientTop  || 0);
		}

		// Add which for key events
		if ( event.which == null && (event.charCode != null || event.keyCode != null) ) {
			event.which = event.charCode != null ? event.charCode : event.keyCode;
		}

		// Add metaKey to non-Mac browsers (use ctrl for PC's and Meta for Macs)
		if ( !event.metaKey && event.ctrlKey ) {
			event.metaKey = event.ctrlKey;
		}

		// Add which for click: 1 === left; 2 === middle; 3 === right
		// Note: button is not normalized, so don't use it
		if ( !event.which && event.button !== undefined ) {
			event.which = (event.button & 1 ? 1 : ( event.button & 2 ? 3 : ( event.button & 4 ? 2 : 0 ) ));
		}

		return event;
	},

	// Deprecated, use jQuery.guid instead
	guid: 1E8,

	// Deprecated, use jQuery.proxy instead
	proxy: jQuery.proxy,

	special: {
		ready: {
			// Make sure the ready event is setup
			setup: jQuery.bindReady,
			teardown: jQuery.noop
		},

		live: {
			add: function( handleObj ) {
				jQuery.event.add( this,
					liveConvert( handleObj.origType, handleObj.selector ),
					jQuery.extend({}, handleObj, {handler: liveHandler, guid: handleObj.handler.guid}) );
			},

			remove: function( handleObj ) {
				jQuery.event.remove( this, liveConvert( handleObj.origType, handleObj.selector ), handleObj );
			}
		},

		beforeunload: {
			setup: function( data, namespaces, eventHandle ) {
				// We only want to do this special case on windows
				if ( jQuery.isWindow( this ) ) {
					this.onbeforeunload = eventHandle;
				}
			},

			teardown: function( namespaces, eventHandle ) {
				if ( this.onbeforeunload === eventHandle ) {
					this.onbeforeunload = null;
				}
			}
		}
	}
};

jQuery.removeEvent = document.removeEventListener ?
	function( elem, type, handle ) {
		if ( elem.removeEventListener ) {
			elem.removeEventListener( type, handle, false );
		}
	} :
	function( elem, type, handle ) {
		if ( elem.detachEvent ) {
			elem.detachEvent( "on" + type, handle );
		}
	};

jQuery.Event = function( src, props ) {
	// Allow instantiation without the 'new' keyword
	if ( !this.preventDefault ) {
		return new jQuery.Event( src, props );
	}

	// Event object
	if ( src && src.type ) {
		this.originalEvent = src;
		this.type = src.type;

		// Events bubbling up the document may have been marked as prevented
		// by a handler lower down the tree; reflect the correct value.
		this.isDefaultPrevented = (src.defaultPrevented || src.returnValue === false ||
			src.getPreventDefault && src.getPreventDefault()) ? returnTrue : returnFalse;

	// Event type
	} else {
		this.type = src;
	}

	// Put explicitly provided properties onto the event object
	if ( props ) {
		jQuery.extend( this, props );
	}

	// timeStamp is buggy for some events on Firefox(#3843)
	// So we won't rely on the native value
	this.timeStamp = jQuery.now();

	// Mark it as fixed
	this[ jQuery.expando ] = true;
};

function returnFalse() {
	return false;
}
function returnTrue() {
	return true;
}

// jQuery.Event is based on DOM3 Events as specified by the ECMAScript Language Binding
// http://www.w3.org/TR/2003/WD-DOM-Level-3-Events-20030331/ecma-script-binding.html
jQuery.Event.prototype = {
	preventDefault: function() {
		this.isDefaultPrevented = returnTrue;

		var e = this.originalEvent;
		if ( !e ) {
			return;
		}

		// if preventDefault exists run it on the original event
		if ( e.preventDefault ) {
			e.preventDefault();

		// otherwise set the returnValue property of the original event to false (IE)
		} else {
			e.returnValue = false;
		}
	},
	stopPropagation: function() {
		this.isPropagationStopped = returnTrue;

		var e = this.originalEvent;
		if ( !e ) {
			return;
		}
		// if stopPropagation exists run it on the original event
		if ( e.stopPropagation ) {
			e.stopPropagation();
		}
		// otherwise set the cancelBubble property of the original event to true (IE)
		e.cancelBubble = true;
	},
	stopImmediatePropagation: function() {
		this.isImmediatePropagationStopped = returnTrue;
		this.stopPropagation();
	},
	isDefaultPrevented: returnFalse,
	isPropagationStopped: returnFalse,
	isImmediatePropagationStopped: returnFalse
};

// Checks if an event happened on an element within another element
// Used in jQuery.event.special.mouseenter and mouseleave handlers
var withinElement = function( event ) {

	// Check if mouse(over|out) are still within the same parent element
	var related = event.relatedTarget,
		inside = false,
		eventType = event.type;

	event.type = event.data;

	if ( related !== this ) {

		if ( related ) {
			inside = jQuery.contains( this, related );
		}

		if ( !inside ) {

			jQuery.event.handle.apply( this, arguments );

			event.type = eventType;
		}
	}
},

// In case of event delegation, we only need to rename the event.type,
// liveHandler will take care of the rest.
delegate = function( event ) {
	event.type = event.data;
	jQuery.event.handle.apply( this, arguments );
};

// Create mouseenter and mouseleave events
jQuery.each({
	mouseenter: "mouseover",
	mouseleave: "mouseout"
}, function( orig, fix ) {
	jQuery.event.special[ orig ] = {
		setup: function( data ) {
			jQuery.event.add( this, fix, data && data.selector ? delegate : withinElement, orig );
		},
		teardown: function( data ) {
			jQuery.event.remove( this, fix, data && data.selector ? delegate : withinElement );
		}
	};
});

// submit delegation
if ( !jQuery.support.submitBubbles ) {

	jQuery.event.special.submit = {
		setup: function( data, namespaces ) {
			if ( !jQuery.nodeName( this, "form" ) ) {
				jQuery.event.add(this, "click.specialSubmit", function( e ) {
					var elem = e.target,
						type = jQuery.nodeName( elem, "input" ) ? elem.type : "";

					if ( (type === "submit" || type === "image") && jQuery( elem ).closest("form").length ) {
						trigger( "submit", this, arguments );
					}
				});

				jQuery.event.add(this, "keypress.specialSubmit", function( e ) {
					var elem = e.target,
						type = jQuery.nodeName( elem, "input" ) ? elem.type : "";

					if ( (type === "text" || type === "password") && jQuery( elem ).closest("form").length && e.keyCode === 13 ) {
						trigger( "submit", this, arguments );
					}
				});

			} else {
				return false;
			}
		},

		teardown: function( namespaces ) {
			jQuery.event.remove( this, ".specialSubmit" );
		}
	};

}

// change delegation, happens here so we have bind.
if ( !jQuery.support.changeBubbles ) {

	var changeFilters,

	getVal = function( elem ) {
		var type = jQuery.nodeName( elem, "input" ) ? elem.type : "",
			val = elem.value;

		if ( type === "radio" || type === "checkbox" ) {
			val = elem.checked;

		} else if ( type === "select-multiple" ) {
			val = elem.selectedIndex > -1 ?
				jQuery.map( elem.options, function( elem ) {
					return elem.selected;
				}).join("-") :
				"";

		} else if ( jQuery.nodeName( elem, "select" ) ) {
			val = elem.selectedIndex;
		}

		return val;
	},

	testChange = function testChange( e ) {
		var elem = e.target, data, val;

		if ( !rformElems.test( elem.nodeName ) || elem.readOnly ) {
			return;
		}

		data = jQuery._data( elem, "_change_data" );
		val = getVal(elem);

		// the current data will be also retrieved by beforeactivate
		if ( e.type !== "focusout" || elem.type !== "radio" ) {
			jQuery._data( elem, "_change_data", val );
		}

		if ( data === undefined || val === data ) {
			return;
		}

		if ( data != null || val ) {
			e.type = "change";
			e.liveFired = undefined;
			jQuery.event.trigger( e, arguments[1], elem );
		}
	};

	jQuery.event.special.change = {
		filters: {
			focusout: testChange,

			beforedeactivate: testChange,

			click: function( e ) {
				var elem = e.target, type = jQuery.nodeName( elem, "input" ) ? elem.type : "";

				if ( type === "radio" || type === "checkbox" || jQuery.nodeName( elem, "select" ) ) {
					testChange.call( this, e );
				}
			},

			// Change has to be called before submit
			// Keydown will be called before keypress, which is used in submit-event delegation
			keydown: function( e ) {
				var elem = e.target, type = jQuery.nodeName( elem, "input" ) ? elem.type : "";

				if ( (e.keyCode === 13 && !jQuery.nodeName( elem, "textarea" ) ) ||
					(e.keyCode === 32 && (type === "checkbox" || type === "radio")) ||
					type === "select-multiple" ) {
					testChange.call( this, e );
				}
			},

			// Beforeactivate happens also before the previous element is blurred
			// with this event you can't trigger a change event, but you can store
			// information
			beforeactivate: function( e ) {
				var elem = e.target;
				jQuery._data( elem, "_change_data", getVal(elem) );
			}
		},

		setup: function( data, namespaces ) {
			if ( this.type === "file" ) {
				return false;
			}

			for ( var type in changeFilters ) {
				jQuery.event.add( this, type + ".specialChange", changeFilters[type] );
			}

			return rformElems.test( this.nodeName );
		},

		teardown: function( namespaces ) {
			jQuery.event.remove( this, ".specialChange" );

			return rformElems.test( this.nodeName );
		}
	};

	changeFilters = jQuery.event.special.change.filters;

	// Handle when the input is .focus()'d
	changeFilters.focus = changeFilters.beforeactivate;
}

function trigger( type, elem, args ) {
	// Piggyback on a donor event to simulate a different one.
	// Fake originalEvent to avoid donor's stopPropagation, but if the
	// simulated event prevents default then we do the same on the donor.
	// Don't pass args or remember liveFired; they apply to the donor event.
	var event = jQuery.extend( {}, args[ 0 ] );
	event.type = type;
	event.originalEvent = {};
	event.liveFired = undefined;
	jQuery.event.handle.call( elem, event );
	if ( event.isDefaultPrevented() ) {
		args[ 0 ].preventDefault();
	}
}

// Create "bubbling" focus and blur events
if ( !jQuery.support.focusinBubbles ) {
	jQuery.each({ focus: "focusin", blur: "focusout" }, function( orig, fix ) {

		// Attach a single capturing handler while someone wants focusin/focusout
		var attaches = 0;

		jQuery.event.special[ fix ] = {
			setup: function() {
				if ( attaches++ === 0 ) {
					document.addEventListener( orig, handler, true );
				}
			},
			teardown: function() {
				if ( --attaches === 0 ) {
					document.removeEventListener( orig, handler, true );
				}
			}
		};

		function handler( donor ) {
			// Donor event is always a native one; fix it and switch its type.
			// Let focusin/out handler cancel the donor focus/blur event.
			var e = jQuery.event.fix( donor );
			e.type = fix;
			e.originalEvent = {};
			jQuery.event.trigger( e, null, e.target );
			if ( e.isDefaultPrevented() ) {
				donor.preventDefault();
			}
		}
	});
}

jQuery.each(["bind", "one"], function( i, name ) {
	jQuery.fn[ name ] = function( type, data, fn ) {
		var handler;

		// Handle object literals
		if ( typeof type === "object" ) {
			for ( var key in type ) {
				this[ name ](key, data, type[key], fn);
			}
			return this;
		}

		if ( arguments.length === 2 || data === false ) {
			fn = data;
			data = undefined;
		}

		if ( name === "one" ) {
			handler = function( event ) {
				jQuery( this ).unbind( event, handler );
				return fn.apply( this, arguments );
			};
			handler.guid = fn.guid || jQuery.guid++;
		} else {
			handler = fn;
		}

		if ( type === "unload" && name !== "one" ) {
			this.one( type, data, fn );

		} else {
			for ( var i = 0, l = this.length; i < l; i++ ) {
				jQuery.event.add( this[i], type, handler, data );
			}
		}

		return this;
	};
});

jQuery.fn.extend({
	unbind: function( type, fn ) {
		// Handle object literals
		if ( typeof type === "object" && !type.preventDefault ) {
			for ( var key in type ) {
				this.unbind(key, type[key]);
			}

		} else {
			for ( var i = 0, l = this.length; i < l; i++ ) {
				jQuery.event.remove( this[i], type, fn );
			}
		}

		return this;
	},

	delegate: function( selector, types, data, fn ) {
		return this.live( types, data, fn, selector );
	},

	undelegate: function( selector, types, fn ) {
		if ( arguments.length === 0 ) {
			return this.unbind( "live" );

		} else {
			return this.die( types, null, fn, selector );
		}
	},

	trigger: function( type, data ) {
		return this.each(function() {
			jQuery.event.trigger( type, data, this );
		});
	},

	triggerHandler: function( type, data ) {
		if ( this[0] ) {
			return jQuery.event.trigger( type, data, this[0], true );
		}
	},

	toggle: function( fn ) {
		// Save reference to arguments for access in closure
		var args = arguments,
			guid = fn.guid || jQuery.guid++,
			i = 0,
			toggler = function( event ) {
				// Figure out which function to execute
				var lastToggle = ( jQuery.data( this, "lastToggle" + fn.guid ) || 0 ) % i;
				jQuery.data( this, "lastToggle" + fn.guid, lastToggle + 1 );

				// Make sure that clicks stop
				event.preventDefault();

				// and execute the function
				return args[ lastToggle ].apply( this, arguments ) || false;
			};

		// link all the functions, so any of them can unbind this click handler
		toggler.guid = guid;
		while ( i < args.length ) {
			args[ i++ ].guid = guid;
		}

		return this.click( toggler );
	},

	hover: function( fnOver, fnOut ) {
		return this.mouseenter( fnOver ).mouseleave( fnOut || fnOver );
	}
});

var liveMap = {
	focus: "focusin",
	blur: "focusout",
	mouseenter: "mouseover",
	mouseleave: "mouseout"
};

jQuery.each(["live", "die"], function( i, name ) {
	jQuery.fn[ name ] = function( types, data, fn, origSelector /* Internal Use Only */ ) {
		var type, i = 0, match, namespaces, preType,
			selector = origSelector || this.selector,
			context = origSelector ? this : jQuery( this.context );

		if ( typeof types === "object" && !types.preventDefault ) {
			for ( var key in types ) {
				context[ name ]( key, data, types[key], selector );
			}

			return this;
		}

		if ( name === "die" && !types &&
					origSelector && origSelector.charAt(0) === "." ) {

			context.unbind( origSelector );

			return this;
		}

		if ( data === false || jQuery.isFunction( data ) ) {
			fn = data || returnFalse;
			data = undefined;
		}

		types = (types || "").split(" ");

		while ( (type = types[ i++ ]) != null ) {
			match = rnamespaces.exec( type );
			namespaces = "";

			if ( match )  {
				namespaces = match[0];
				type = type.replace( rnamespaces, "" );
			}

			if ( type === "hover" ) {
				types.push( "mouseenter" + namespaces, "mouseleave" + namespaces );
				continue;
			}

			preType = type;

			if ( liveMap[ type ] ) {
				types.push( liveMap[ type ] + namespaces );
				type = type + namespaces;

			} else {
				type = (liveMap[ type ] || type) + namespaces;
			}

			if ( name === "live" ) {
				// bind live handler
				for ( var j = 0, l = context.length; j < l; j++ ) {
					jQuery.event.add( context[j], "live." + liveConvert( type, selector ),
						{ data: data, selector: selector, handler: fn, origType: type, origHandler: fn, preType: preType } );
				}

			} else {
				// unbind live handler
				context.unbind( "live." + liveConvert( type, selector ), fn );
			}
		}

		return this;
	};
});

function liveHandler( event ) {
	var stop, maxLevel, related, match, handleObj, elem, j, i, l, data, close, namespace, ret,
		elems = [],
		selectors = [],
		events = jQuery._data( this, "events" );

	// Make sure we avoid non-left-click bubbling in Firefox (#3861) and disabled elements in IE (#6911)
	if ( event.liveFired === this || !events || !events.live || event.target.disabled || event.button && event.type === "click" ) {
		return;
	}

	if ( event.namespace ) {
		namespace = new RegExp("(^|\\.)" + event.namespace.split(".").join("\\.(?:.*\\.)?") + "(\\.|$)");
	}

	event.liveFired = this;

	var live = events.live.slice(0);

	for ( j = 0; j < live.length; j++ ) {
		handleObj = live[j];

		if ( handleObj.origType.replace( rnamespaces, "" ) === event.type ) {
			selectors.push( handleObj.selector );

		} else {
			live.splice( j--, 1 );
		}
	}

	match = jQuery( event.target ).closest( selectors, event.currentTarget );

	for ( i = 0, l = match.length; i < l; i++ ) {
		close = match[i];

		for ( j = 0; j < live.length; j++ ) {
			handleObj = live[j];

			if ( close.selector === handleObj.selector && (!namespace || namespace.test( handleObj.namespace )) && !close.elem.disabled ) {
				elem = close.elem;
				related = null;

				// Those two events require additional checking
				if ( handleObj.preType === "mouseenter" || handleObj.preType === "mouseleave" ) {
					event.type = handleObj.preType;
					related = jQuery( event.relatedTarget ).closest( handleObj.selector )[0];

					// Make sure not to accidentally match a child element with the same selector
					if ( related && jQuery.contains( elem, related ) ) {
						related = elem;
					}
				}

				if ( !related || related !== elem ) {
					elems.push({ elem: elem, handleObj: handleObj, level: close.level });
				}
			}
		}
	}

	for ( i = 0, l = elems.length; i < l; i++ ) {
		match = elems[i];

		if ( maxLevel && match.level > maxLevel ) {
			break;
		}

		event.currentTarget = match.elem;
		event.data = match.handleObj.data;
		event.handleObj = match.handleObj;

		ret = match.handleObj.origHandler.apply( match.elem, arguments );

		if ( ret === false || event.isPropagationStopped() ) {
			maxLevel = match.level;

			if ( ret === false ) {
				stop = false;
			}
			if ( event.isImmediatePropagationStopped() ) {
				break;
			}
		}
	}

	return stop;
}

function liveConvert( type, selector ) {
	return (type && type !== "*" ? type + "." : "") + selector.replace(rperiod, "`").replace(rspaces, "&");
}

jQuery.each( ("blur focus focusin focusout load resize scroll unload click dblclick " +
	"mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave " +
	"change select submit keydown keypress keyup error").split(" "), function( i, name ) {

	// Handle event binding
	jQuery.fn[ name ] = function( data, fn ) {
		if ( fn == null ) {
			fn = data;
			data = null;
		}

		return arguments.length > 0 ?
			this.bind( name, data, fn ) :
			this.trigger( name );
	};

	if ( jQuery.attrFn ) {
		jQuery.attrFn[ name ] = true;
	}
});



/*!
 * Sizzle CSS Selector Engine
 *  Copyright 2011, The Dojo Foundation
 *  Released under the MIT, BSD, and GPL Licenses.
 *  More information: http://sizzlejs.com/
 */
(function(){

var chunker = /((?:\((?:\([^()]+\)|[^()]+)+\)|\[(?:\[[^\[\]]*\]|['"][^'"]*['"]|[^\[\]'"]+)+\]|\\.|[^ >+~,(\[\\]+)+|[>+~])(\s*,\s*)?((?:.|\r|\n)*)/g,
	done = 0,
	toString = Object.prototype.toString,
	hasDuplicate = false,
	baseHasDuplicate = true,
	rBackslash = /\\/g,
	rNonWord = /\W/;

// Here we check if the JavaScript engine is using some sort of
// optimization where it does not always call our comparision
// function. If that is the case, discard the hasDuplicate value.
//   Thus far that includes Google Chrome.
[0, 0].sort(function() {
	baseHasDuplicate = false;
	return 0;
});

var Sizzle = function( selector, context, results, seed ) {
	results = results || [];
	context = context || document;

	var origContext = context;

	if ( context.nodeType !== 1 && context.nodeType !== 9 ) {
		return [];
	}
	
	if ( !selector || typeof selector !== "string" ) {
		return results;
	}

	var m, set, checkSet, extra, ret, cur, pop, i,
		prune = true,
		contextXML = Sizzle.isXML( context ),
		parts = [],
		soFar = selector;
	
	// Reset the position of the chunker regexp (start from head)
	do {
		chunker.exec( "" );
		m = chunker.exec( soFar );

		if ( m ) {
			soFar = m[3];
		
			parts.push( m[1] );
		
			if ( m[2] ) {
				extra = m[3];
				break;
			}
		}
	} while ( m );

	if ( parts.length > 1 && origPOS.exec( selector ) ) {

		if ( parts.length === 2 && Expr.relative[ parts[0] ] ) {
			set = posProcess( parts[0] + parts[1], context );

		} else {
			set = Expr.relative[ parts[0] ] ?
				[ context ] :
				Sizzle( parts.shift(), context );

			while ( parts.length ) {
				selector = parts.shift();

				if ( Expr.relative[ selector ] ) {
					selector += parts.shift();
				}
				
				set = posProcess( selector, set );
			}
		}

	} else {
		// Take a shortcut and set the context if the root selector is an ID
		// (but not if it'll be faster if the inner selector is an ID)
		if ( !seed && parts.length > 1 && context.nodeType === 9 && !contextXML &&
				Expr.match.ID.test(parts[0]) && !Expr.match.ID.test(parts[parts.length - 1]) ) {

			ret = Sizzle.find( parts.shift(), context, contextXML );
			context = ret.expr ?
				Sizzle.filter( ret.expr, ret.set )[0] :
				ret.set[0];
		}

		if ( context ) {
			ret = seed ?
				{ expr: parts.pop(), set: makeArray(seed) } :
				Sizzle.find( parts.pop(), parts.length === 1 && (parts[0] === "~" || parts[0] === "+") && context.parentNode ? context.parentNode : context, contextXML );

			set = ret.expr ?
				Sizzle.filter( ret.expr, ret.set ) :
				ret.set;

			if ( parts.length > 0 ) {
				checkSet = makeArray( set );

			} else {
				prune = false;
			}

			while ( parts.length ) {
				cur = parts.pop();
				pop = cur;

				if ( !Expr.relative[ cur ] ) {
					cur = "";
				} else {
					pop = parts.pop();
				}

				if ( pop == null ) {
					pop = context;
				}

				Expr.relative[ cur ]( checkSet, pop, contextXML );
			}

		} else {
			checkSet = parts = [];
		}
	}

	if ( !checkSet ) {
		checkSet = set;
	}

	if ( !checkSet ) {
		Sizzle.error( cur || selector );
	}

	if ( toString.call(checkSet) === "[object Array]" ) {
		if ( !prune ) {
			results.push.apply( results, checkSet );

		} else if ( context && context.nodeType === 1 ) {
			for ( i = 0; checkSet[i] != null; i++ ) {
				if ( checkSet[i] && (checkSet[i] === true || checkSet[i].nodeType === 1 && Sizzle.contains(context, checkSet[i])) ) {
					results.push( set[i] );
				}
			}

		} else {
			for ( i = 0; checkSet[i] != null; i++ ) {
				if ( checkSet[i] && checkSet[i].nodeType === 1 ) {
					results.push( set[i] );
				}
			}
		}

	} else {
		makeArray( checkSet, results );
	}

	if ( extra ) {
		Sizzle( extra, origContext, results, seed );
		Sizzle.uniqueSort( results );
	}

	return results;
};

Sizzle.uniqueSort = function( results ) {
	if ( sortOrder ) {
		hasDuplicate = baseHasDuplicate;
		results.sort( sortOrder );

		if ( hasDuplicate ) {
			for ( var i = 1; i < results.length; i++ ) {
				if ( results[i] === results[ i - 1 ] ) {
					results.splice( i--, 1 );
				}
			}
		}
	}

	return results;
};

Sizzle.matches = function( expr, set ) {
	return Sizzle( expr, null, null, set );
};

Sizzle.matchesSelector = function( node, expr ) {
	return Sizzle( expr, null, null, [node] ).length > 0;
};

Sizzle.find = function( expr, context, isXML ) {
	var set;

	if ( !expr ) {
		return [];
	}

	for ( var i = 0, l = Expr.order.length; i < l; i++ ) {
		var match,
			type = Expr.order[i];
		
		if ( (match = Expr.leftMatch[ type ].exec( expr )) ) {
			var left = match[1];
			match.splice( 1, 1 );

			if ( left.substr( left.length - 1 ) !== "\\" ) {
				match[1] = (match[1] || "").replace( rBackslash, "" );
				set = Expr.find[ type ]( match, context, isXML );

				if ( set != null ) {
					expr = expr.replace( Expr.match[ type ], "" );
					break;
				}
			}
		}
	}

	if ( !set ) {
		set = typeof context.getElementsByTagName !== "undefined" ?
			context.getElementsByTagName( "*" ) :
			[];
	}

	return { set: set, expr: expr };
};

Sizzle.filter = function( expr, set, inplace, not ) {
	var match, anyFound,
		old = expr,
		result = [],
		curLoop = set,
		isXMLFilter = set && set[0] && Sizzle.isXML( set[0] );

	while ( expr && set.length ) {
		for ( var type in Expr.filter ) {
			if ( (match = Expr.leftMatch[ type ].exec( expr )) != null && match[2] ) {
				var found, item,
					filter = Expr.filter[ type ],
					left = match[1];

				anyFound = false;

				match.splice(1,1);

				if ( left.substr( left.length - 1 ) === "\\" ) {
					continue;
				}

				if ( curLoop === result ) {
					result = [];
				}

				if ( Expr.preFilter[ type ] ) {
					match = Expr.preFilter[ type ]( match, curLoop, inplace, result, not, isXMLFilter );

					if ( !match ) {
						anyFound = found = true;

					} else if ( match === true ) {
						continue;
					}
				}

				if ( match ) {
					for ( var i = 0; (item = curLoop[i]) != null; i++ ) {
						if ( item ) {
							found = filter( item, match, i, curLoop );
							var pass = not ^ !!found;

							if ( inplace && found != null ) {
								if ( pass ) {
									anyFound = true;

								} else {
									curLoop[i] = false;
								}

							} else if ( pass ) {
								result.push( item );
								anyFound = true;
							}
						}
					}
				}

				if ( found !== undefined ) {
					if ( !inplace ) {
						curLoop = result;
					}

					expr = expr.replace( Expr.match[ type ], "" );

					if ( !anyFound ) {
						return [];
					}

					break;
				}
			}
		}

		// Improper expression
		if ( expr === old ) {
			if ( anyFound == null ) {
				Sizzle.error( expr );

			} else {
				break;
			}
		}

		old = expr;
	}

	return curLoop;
};

Sizzle.error = function( msg ) {
	throw "Syntax error, unrecognized expression: " + msg;
};

var Expr = Sizzle.selectors = {
	order: [ "ID", "NAME", "TAG" ],

	match: {
		ID: /#((?:[\w\u00c0-\uFFFF\-]|\\.)+)/,
		CLASS: /\.((?:[\w\u00c0-\uFFFF\-]|\\.)+)/,
		NAME: /\[name=['"]*((?:[\w\u00c0-\uFFFF\-]|\\.)+)['"]*\]/,
		ATTR: /\[\s*((?:[\w\u00c0-\uFFFF\-]|\\.)+)\s*(?:(\S?=)\s*(?:(['"])(.*?)\3|(#?(?:[\w\u00c0-\uFFFF\-]|\\.)*)|)|)\s*\]/,
		TAG: /^((?:[\w\u00c0-\uFFFF\*\-]|\\.)+)/,
		CHILD: /:(only|nth|last|first)-child(?:\(\s*(even|odd|(?:[+\-]?\d+|(?:[+\-]?\d*)?n\s*(?:[+\-]\s*\d+)?))\s*\))?/,
		POS: /:(nth|eq|gt|lt|first|last|even|odd)(?:\((\d*)\))?(?=[^\-]|$)/,
		PSEUDO: /:((?:[\w\u00c0-\uFFFF\-]|\\.)+)(?:\((['"]?)((?:\([^\)]+\)|[^\(\)]*)+)\2\))?/
	},

	leftMatch: {},

	attrMap: {
		"class": "className",
		"for": "htmlFor"
	},

	attrHandle: {
		href: function( elem ) {
			return elem.getAttribute( "href" );
		},
		type: function( elem ) {
			return elem.getAttribute( "type" );
		}
	},

	relative: {
		"+": function(checkSet, part){
			var isPartStr = typeof part === "string",
				isTag = isPartStr && !rNonWord.test( part ),
				isPartStrNotTag = isPartStr && !isTag;

			if ( isTag ) {
				part = part.toLowerCase();
			}

			for ( var i = 0, l = checkSet.length, elem; i < l; i++ ) {
				if ( (elem = checkSet[i]) ) {
					while ( (elem = elem.previousSibling) && elem.nodeType !== 1 ) {}

					checkSet[i] = isPartStrNotTag || elem && elem.nodeName.toLowerCase() === part ?
						elem || false :
						elem === part;
				}
			}

			if ( isPartStrNotTag ) {
				Sizzle.filter( part, checkSet, true );
			}
		},

		">": function( checkSet, part ) {
			var elem,
				isPartStr = typeof part === "string",
				i = 0,
				l = checkSet.length;

			if ( isPartStr && !rNonWord.test( part ) ) {
				part = part.toLowerCase();

				for ( ; i < l; i++ ) {
					elem = checkSet[i];

					if ( elem ) {
						var parent = elem.parentNode;
						checkSet[i] = parent.nodeName.toLowerCase() === part ? parent : false;
					}
				}

			} else {
				for ( ; i < l; i++ ) {
					elem = checkSet[i];

					if ( elem ) {
						checkSet[i] = isPartStr ?
							elem.parentNode :
							elem.parentNode === part;
					}
				}

				if ( isPartStr ) {
					Sizzle.filter( part, checkSet, true );
				}
			}
		},

		"": function(checkSet, part, isXML){
			var nodeCheck,
				doneName = done++,
				checkFn = dirCheck;

			if ( typeof part === "string" && !rNonWord.test( part ) ) {
				part = part.toLowerCase();
				nodeCheck = part;
				checkFn = dirNodeCheck;
			}

			checkFn( "parentNode", part, doneName, checkSet, nodeCheck, isXML );
		},

		"~": function( checkSet, part, isXML ) {
			var nodeCheck,
				doneName = done++,
				checkFn = dirCheck;

			if ( typeof part === "string" && !rNonWord.test( part ) ) {
				part = part.toLowerCase();
				nodeCheck = part;
				checkFn = dirNodeCheck;
			}

			checkFn( "previousSibling", part, doneName, checkSet, nodeCheck, isXML );
		}
	},

	find: {
		ID: function( match, context, isXML ) {
			if ( typeof context.getElementById !== "undefined" && !isXML ) {
				var m = context.getElementById(match[1]);
				// Check parentNode to catch when Blackberry 4.6 returns
				// nodes that are no longer in the document #6963
				return m && m.parentNode ? [m] : [];
			}
		},

		NAME: function( match, context ) {
			if ( typeof context.getElementsByName !== "undefined" ) {
				var ret = [],
					results = context.getElementsByName( match[1] );

				for ( var i = 0, l = results.length; i < l; i++ ) {
					if ( results[i].getAttribute("name") === match[1] ) {
						ret.push( results[i] );
					}
				}

				return ret.length === 0 ? null : ret;
			}
		},

		TAG: function( match, context ) {
			if ( typeof context.getElementsByTagName !== "undefined" ) {
				return context.getElementsByTagName( match[1] );
			}
		}
	},
	preFilter: {
		CLASS: function( match, curLoop, inplace, result, not, isXML ) {
			match = " " + match[1].replace( rBackslash, "" ) + " ";

			if ( isXML ) {
				return match;
			}

			for ( var i = 0, elem; (elem = curLoop[i]) != null; i++ ) {
				if ( elem ) {
					if ( not ^ (elem.className && (" " + elem.className + " ").replace(/[\t\n\r]/g, " ").indexOf(match) >= 0) ) {
						if ( !inplace ) {
							result.push( elem );
						}

					} else if ( inplace ) {
						curLoop[i] = false;
					}
				}
			}

			return false;
		},

		ID: function( match ) {
			return match[1].replace( rBackslash, "" );
		},

		TAG: function( match, curLoop ) {
			return match[1].replace( rBackslash, "" ).toLowerCase();
		},

		CHILD: function( match ) {
			if ( match[1] === "nth" ) {
				if ( !match[2] ) {
					Sizzle.error( match[0] );
				}

				match[2] = match[2].replace(/^\+|\s*/g, '');

				// parse equations like 'even', 'odd', '5', '2n', '3n+2', '4n-1', '-n+6'
				var test = /(-?)(\d*)(?:n([+\-]?\d*))?/.exec(
					match[2] === "even" && "2n" || match[2] === "odd" && "2n+1" ||
					!/\D/.test( match[2] ) && "0n+" + match[2] || match[2]);

				// calculate the numbers (first)n+(last) including if they are negative
				match[2] = (test[1] + (test[2] || 1)) - 0;
				match[3] = test[3] - 0;
			}
			else if ( match[2] ) {
				Sizzle.error( match[0] );
			}

			// TODO: Move to normal caching system
			match[0] = done++;

			return match;
		},

		ATTR: function( match, curLoop, inplace, result, not, isXML ) {
			var name = match[1] = match[1].replace( rBackslash, "" );
			
			if ( !isXML && Expr.attrMap[name] ) {
				match[1] = Expr.attrMap[name];
			}

			// Handle if an un-quoted value was used
			match[4] = ( match[4] || match[5] || "" ).replace( rBackslash, "" );

			if ( match[2] === "~=" ) {
				match[4] = " " + match[4] + " ";
			}

			return match;
		},

		PSEUDO: function( match, curLoop, inplace, result, not ) {
			if ( match[1] === "not" ) {
				// If we're dealing with a complex expression, or a simple one
				if ( ( chunker.exec(match[3]) || "" ).length > 1 || /^\w/.test(match[3]) ) {
					match[3] = Sizzle(match[3], null, null, curLoop);

				} else {
					var ret = Sizzle.filter(match[3], curLoop, inplace, true ^ not);

					if ( !inplace ) {
						result.push.apply( result, ret );
					}

					return false;
				}

			} else if ( Expr.match.POS.test( match[0] ) || Expr.match.CHILD.test( match[0] ) ) {
				return true;
			}
			
			return match;
		},

		POS: function( match ) {
			match.unshift( true );

			return match;
		}
	},
	
	filters: {
		enabled: function( elem ) {
			return elem.disabled === false && elem.type !== "hidden";
		},

		disabled: function( elem ) {
			return elem.disabled === true;
		},

		checked: function( elem ) {
			return elem.checked === true;
		},
		
		selected: function( elem ) {
			// Accessing this property makes selected-by-default
			// options in Safari work properly
			if ( elem.parentNode ) {
				elem.parentNode.selectedIndex;
			}
			
			return elem.selected === true;
		},

		parent: function( elem ) {
			return !!elem.firstChild;
		},

		empty: function( elem ) {
			return !elem.firstChild;
		},

		has: function( elem, i, match ) {
			return !!Sizzle( match[3], elem ).length;
		},

		header: function( elem ) {
			return (/h\d/i).test( elem.nodeName );
		},

		text: function( elem ) {
			var attr = elem.getAttribute( "type" ), type = elem.type;
			// IE6 and 7 will map elem.type to 'text' for new HTML5 types (search, etc) 
			// use getAttribute instead to test this case
			return elem.nodeName.toLowerCase() === "input" && "text" === type && ( attr === type || attr === null );
		},

		radio: function( elem ) {
			return elem.nodeName.toLowerCase() === "input" && "radio" === elem.type;
		},

		checkbox: function( elem ) {
			return elem.nodeName.toLowerCase() === "input" && "checkbox" === elem.type;
		},

		file: function( elem ) {
			return elem.nodeName.toLowerCase() === "input" && "file" === elem.type;
		},

		password: function( elem ) {
			return elem.nodeName.toLowerCase() === "input" && "password" === elem.type;
		},

		submit: function( elem ) {
			var name = elem.nodeName.toLowerCase();
			return (name === "input" || name === "button") && "submit" === elem.type;
		},

		image: function( elem ) {
			return elem.nodeName.toLowerCase() === "input" && "image" === elem.type;
		},

		reset: function( elem ) {
			var name = elem.nodeName.toLowerCase();
			return (name === "input" || name === "button") && "reset" === elem.type;
		},

		button: function( elem ) {
			var name = elem.nodeName.toLowerCase();
			return name === "input" && "button" === elem.type || name === "button";
		},

		input: function( elem ) {
			return (/input|select|textarea|button/i).test( elem.nodeName );
		},

		focus: function( elem ) {
			return elem === elem.ownerDocument.activeElement;
		}
	},
	setFilters: {
		first: function( elem, i ) {
			return i === 0;
		},

		last: function( elem, i, match, array ) {
			return i === array.length - 1;
		},

		even: function( elem, i ) {
			return i % 2 === 0;
		},

		odd: function( elem, i ) {
			return i % 2 === 1;
		},

		lt: function( elem, i, match ) {
			return i < match[3] - 0;
		},

		gt: function( elem, i, match ) {
			return i > match[3] - 0;
		},

		nth: function( elem, i, match ) {
			return match[3] - 0 === i;
		},

		eq: function( elem, i, match ) {
			return match[3] - 0 === i;
		}
	},
	filter: {
		PSEUDO: function( elem, match, i, array ) {
			var name = match[1],
				filter = Expr.filters[ name ];

			if ( filter ) {
				return filter( elem, i, match, array );

			} else if ( name === "contains" ) {
				return (elem.textContent || elem.innerText || Sizzle.getText([ elem ]) || "").indexOf(match[3]) >= 0;

			} else if ( name === "not" ) {
				var not = match[3];

				for ( var j = 0, l = not.length; j < l; j++ ) {
					if ( not[j] === elem ) {
						return false;
					}
				}

				return true;

			} else {
				Sizzle.error( name );
			}
		},

		CHILD: function( elem, match ) {
			var type = match[1],
				node = elem;

			switch ( type ) {
				case "only":
				case "first":
					while ( (node = node.previousSibling) )	 {
						if ( node.nodeType === 1 ) { 
							return false; 
						}
					}

					if ( type === "first" ) { 
						return true; 
					}

					node = elem;

				case "last":
					while ( (node = node.nextSibling) )	 {
						if ( node.nodeType === 1 ) { 
							return false; 
						}
					}

					return true;

				case "nth":
					var first = match[2],
						last = match[3];

					if ( first === 1 && last === 0 ) {
						return true;
					}
					
					var doneName = match[0],
						parent = elem.parentNode;
	
					if ( parent && (parent.sizcache !== doneName || !elem.nodeIndex) ) {
						var count = 0;
						
						for ( node = parent.firstChild; node; node = node.nextSibling ) {
							if ( node.nodeType === 1 ) {
								node.nodeIndex = ++count;
							}
						} 

						parent.sizcache = doneName;
					}
					
					var diff = elem.nodeIndex - last;

					if ( first === 0 ) {
						return diff === 0;

					} else {
						return ( diff % first === 0 && diff / first >= 0 );
					}
			}
		},

		ID: function( elem, match ) {
			return elem.nodeType === 1 && elem.getAttribute("id") === match;
		},

		TAG: function( elem, match ) {
			return (match === "*" && elem.nodeType === 1) || elem.nodeName.toLowerCase() === match;
		},
		
		CLASS: function( elem, match ) {
			return (" " + (elem.className || elem.getAttribute("class")) + " ")
				.indexOf( match ) > -1;
		},

		ATTR: function( elem, match ) {
			var name = match[1],
				result = Expr.attrHandle[ name ] ?
					Expr.attrHandle[ name ]( elem ) :
					elem[ name ] != null ?
						elem[ name ] :
						elem.getAttribute( name ),
				value = result + "",
				type = match[2],
				check = match[4];

			return result == null ?
				type === "!=" :
				type === "=" ?
				value === check :
				type === "*=" ?
				value.indexOf(check) >= 0 :
				type === "~=" ?
				(" " + value + " ").indexOf(check) >= 0 :
				!check ?
				value && result !== false :
				type === "!=" ?
				value !== check :
				type === "^=" ?
				value.indexOf(check) === 0 :
				type === "$=" ?
				value.substr(value.length - check.length) === check :
				type === "|=" ?
				value === check || value.substr(0, check.length + 1) === check + "-" :
				false;
		},

		POS: function( elem, match, i, array ) {
			var name = match[2],
				filter = Expr.setFilters[ name ];

			if ( filter ) {
				return filter( elem, i, match, array );
			}
		}
	}
};

var origPOS = Expr.match.POS,
	fescape = function(all, num){
		return "\\" + (num - 0 + 1);
	};

for ( var type in Expr.match ) {
	Expr.match[ type ] = new RegExp( Expr.match[ type ].source + (/(?![^\[]*\])(?![^\(]*\))/.source) );
	Expr.leftMatch[ type ] = new RegExp( /(^(?:.|\r|\n)*?)/.source + Expr.match[ type ].source.replace(/\\(\d+)/g, fescape) );
}

var makeArray = function( array, results ) {
	array = Array.prototype.slice.call( array, 0 );

	if ( results ) {
		results.push.apply( results, array );
		return results;
	}
	
	return array;
};

// Perform a simple check to determine if the browser is capable of
// converting a NodeList to an array using builtin methods.
// Also verifies that the returned array holds DOM nodes
// (which is not the case in the Blackberry browser)
try {
	Array.prototype.slice.call( document.documentElement.childNodes, 0 )[0].nodeType;

// Provide a fallback method if it does not work
} catch( e ) {
	makeArray = function( array, results ) {
		var i = 0,
			ret = results || [];

		if ( toString.call(array) === "[object Array]" ) {
			Array.prototype.push.apply( ret, array );

		} else {
			if ( typeof array.length === "number" ) {
				for ( var l = array.length; i < l; i++ ) {
					ret.push( array[i] );
				}

			} else {
				for ( ; array[i]; i++ ) {
					ret.push( array[i] );
				}
			}
		}

		return ret;
	};
}

var sortOrder, siblingCheck;

if ( document.documentElement.compareDocumentPosition ) {
	sortOrder = function( a, b ) {
		if ( a === b ) {
			hasDuplicate = true;
			return 0;
		}

		if ( !a.compareDocumentPosition || !b.compareDocumentPosition ) {
			return a.compareDocumentPosition ? -1 : 1;
		}

		return a.compareDocumentPosition(b) & 4 ? -1 : 1;
	};

} else {
	sortOrder = function( a, b ) {
		// The nodes are identical, we can exit early
		if ( a === b ) {
			hasDuplicate = true;
			return 0;

		// Fallback to using sourceIndex (in IE) if it's available on both nodes
		} else if ( a.sourceIndex && b.sourceIndex ) {
			return a.sourceIndex - b.sourceIndex;
		}

		var al, bl,
			ap = [],
			bp = [],
			aup = a.parentNode,
			bup = b.parentNode,
			cur = aup;

		// If the nodes are siblings (or identical) we can do a quick check
		if ( aup === bup ) {
			return siblingCheck( a, b );

		// If no parents were found then the nodes are disconnected
		} else if ( !aup ) {
			return -1;

		} else if ( !bup ) {
			return 1;
		}

		// Otherwise they're somewhere else in the tree so we need
		// to build up a full list of the parentNodes for comparison
		while ( cur ) {
			ap.unshift( cur );
			cur = cur.parentNode;
		}

		cur = bup;

		while ( cur ) {
			bp.unshift( cur );
			cur = cur.parentNode;
		}

		al = ap.length;
		bl = bp.length;

		// Start walking down the tree looking for a discrepancy
		for ( var i = 0; i < al && i < bl; i++ ) {
			if ( ap[i] !== bp[i] ) {
				return siblingCheck( ap[i], bp[i] );
			}
		}

		// We ended someplace up the tree so do a sibling check
		return i === al ?
			siblingCheck( a, bp[i], -1 ) :
			siblingCheck( ap[i], b, 1 );
	};

	siblingCheck = function( a, b, ret ) {
		if ( a === b ) {
			return ret;
		}

		var cur = a.nextSibling;

		while ( cur ) {
			if ( cur === b ) {
				return -1;
			}

			cur = cur.nextSibling;
		}

		return 1;
	};
}

// Utility function for retreiving the text value of an array of DOM nodes
Sizzle.getText = function( elems ) {
	var ret = "", elem;

	for ( var i = 0; elems[i]; i++ ) {
		elem = elems[i];

		// Get the text from text nodes and CDATA nodes
		if ( elem.nodeType === 3 || elem.nodeType === 4 ) {
			ret += elem.nodeValue;

		// Traverse everything else, except comment nodes
		} else if ( elem.nodeType !== 8 ) {
			ret += Sizzle.getText( elem.childNodes );
		}
	}

	return ret;
};

// Check to see if the browser returns elements by name when
// querying by getElementById (and provide a workaround)
(function(){
	// We're going to inject a fake input element with a specified name
	var form = document.createElement("div"),
		id = "script" + (new Date()).getTime(),
		root = document.documentElement;

	form.innerHTML = "<a name='" + id + "'/>";

	// Inject it into the root element, check its status, and remove it quickly
	root.insertBefore( form, root.firstChild );

	// The workaround has to do additional checks after a getElementById
	// Which slows things down for other browsers (hence the branching)
	if ( document.getElementById( id ) ) {
		Expr.find.ID = function( match, context, isXML ) {
			if ( typeof context.getElementById !== "undefined" && !isXML ) {
				var m = context.getElementById(match[1]);

				return m ?
					m.id === match[1] || typeof m.getAttributeNode !== "undefined" && m.getAttributeNode("id").nodeValue === match[1] ?
						[m] :
						undefined :
					[];
			}
		};

		Expr.filter.ID = function( elem, match ) {
			var node = typeof elem.getAttributeNode !== "undefined" && elem.getAttributeNode("id");

			return elem.nodeType === 1 && node && node.nodeValue === match;
		};
	}

	root.removeChild( form );

	// release memory in IE
	root = form = null;
})();

(function(){
	// Check to see if the browser returns only elements
	// when doing getElementsByTagName("*")

	// Create a fake element
	var div = document.createElement("div");
	div.appendChild( document.createComment("") );

	// Make sure no comments are found
	if ( div.getElementsByTagName("*").length > 0 ) {
		Expr.find.TAG = function( match, context ) {
			var results = context.getElementsByTagName( match[1] );

			// Filter out possible comments
			if ( match[1] === "*" ) {
				var tmp = [];

				for ( var i = 0; results[i]; i++ ) {
					if ( results[i].nodeType === 1 ) {
						tmp.push( results[i] );
					}
				}

				results = tmp;
			}

			return results;
		};
	}

	// Check to see if an attribute returns normalized href attributes
	div.innerHTML = "<a href='#'></a>";

	if ( div.firstChild && typeof div.firstChild.getAttribute !== "undefined" &&
			div.firstChild.getAttribute("href") !== "#" ) {

		Expr.attrHandle.href = function( elem ) {
			return elem.getAttribute( "href", 2 );
		};
	}

	// release memory in IE
	div = null;
})();

if ( document.querySelectorAll ) {
	(function(){
		var oldSizzle = Sizzle,
			div = document.createElement("div"),
			id = "__sizzle__";

		div.innerHTML = "<p class='TEST'></p>";

		// Safari can't handle uppercase or unicode characters when
		// in quirks mode.
		if ( div.querySelectorAll && div.querySelectorAll(".TEST").length === 0 ) {
			return;
		}
	
		Sizzle = function( query, context, extra, seed ) {
			context = context || document;

			// Only use querySelectorAll on non-XML documents
			// (ID selectors don't work in non-HTML documents)
			if ( !seed && !Sizzle.isXML(context) ) {
				// See if we find a selector to speed up
				var match = /^(\w+$)|^\.([\w\-]+$)|^#([\w\-]+$)/.exec( query );
				
				if ( match && (context.nodeType === 1 || context.nodeType === 9) ) {
					// Speed-up: Sizzle("TAG")
					if ( match[1] ) {
						return makeArray( context.getElementsByTagName( query ), extra );
					
					// Speed-up: Sizzle(".CLASS")
					} else if ( match[2] && Expr.find.CLASS && context.getElementsByClassName ) {
						return makeArray( context.getElementsByClassName( match[2] ), extra );
					}
				}
				
				if ( context.nodeType === 9 ) {
					// Speed-up: Sizzle("body")
					// The body element only exists once, optimize finding it
					if ( query === "body" && context.body ) {
						return makeArray( [ context.body ], extra );
						
					// Speed-up: Sizzle("#ID")
					} else if ( match && match[3] ) {
						var elem = context.getElementById( match[3] );

						// Check parentNode to catch when Blackberry 4.6 returns
						// nodes that are no longer in the document #6963
						if ( elem && elem.parentNode ) {
							// Handle the case where IE and Opera return items
							// by name instead of ID
							if ( elem.id === match[3] ) {
								return makeArray( [ elem ], extra );
							}
							
						} else {
							return makeArray( [], extra );
						}
					}
					
					try {
						return makeArray( context.querySelectorAll(query), extra );
					} catch(qsaError) {}

				// qSA works strangely on Element-rooted queries
				// We can work around this by specifying an extra ID on the root
				// and working up from there (Thanks to Andrew Dupont for the technique)
				// IE 8 doesn't work on object elements
				} else if ( context.nodeType === 1 && context.nodeName.toLowerCase() !== "object" ) {
					var oldContext = context,
						old = context.getAttribute( "id" ),
						nid = old || id,
						hasParent = context.parentNode,
						relativeHierarchySelector = /^\s*[+~]/.test( query );

					if ( !old ) {
						context.setAttribute( "id", nid );
					} else {
						nid = nid.replace( /'/g, "\\$&" );
					}
					if ( relativeHierarchySelector && hasParent ) {
						context = context.parentNode;
					}

					try {
						if ( !relativeHierarchySelector || hasParent ) {
							return makeArray( context.querySelectorAll( "[id='" + nid + "'] " + query ), extra );
						}

					} catch(pseudoError) {
					} finally {
						if ( !old ) {
							oldContext.removeAttribute( "id" );
						}
					}
				}
			}
		
			return oldSizzle(query, context, extra, seed);
		};

		for ( var prop in oldSizzle ) {
			Sizzle[ prop ] = oldSizzle[ prop ];
		}

		// release memory in IE
		div = null;
	})();
}

(function(){
	var html = document.documentElement,
		matches = html.matchesSelector || html.mozMatchesSelector || html.webkitMatchesSelector || html.msMatchesSelector;

	if ( matches ) {
		// Check to see if it's possible to do matchesSelector
		// on a disconnected node (IE 9 fails this)
		var disconnectedMatch = !matches.call( document.createElement( "div" ), "div" ),
			pseudoWorks = false;

		try {
			// This should fail with an exception
			// Gecko does not error, returns false instead
			matches.call( document.documentElement, "[test!='']:sizzle" );
	
		} catch( pseudoError ) {
			pseudoWorks = true;
		}

		Sizzle.matchesSelector = function( node, expr ) {
			// Make sure that attribute selectors are quoted
			expr = expr.replace(/\=\s*([^'"\]]*)\s*\]/g, "='$1']");

			if ( !Sizzle.isXML( node ) ) {
				try { 
					if ( pseudoWorks || !Expr.match.PSEUDO.test( expr ) && !/!=/.test( expr ) ) {
						var ret = matches.call( node, expr );

						// IE 9's matchesSelector returns false on disconnected nodes
						if ( ret || !disconnectedMatch ||
								// As well, disconnected nodes are said to be in a document
								// fragment in IE 9, so check for that
								node.document && node.document.nodeType !== 11 ) {
							return ret;
						}
					}
				} catch(e) {}
			}

			return Sizzle(expr, null, null, [node]).length > 0;
		};
	}
})();

(function(){
	var div = document.createElement("div");

	div.innerHTML = "<div class='test e'></div><div class='test'></div>";

	// Opera can't find a second classname (in 9.6)
	// Also, make sure that getElementsByClassName actually exists
	if ( !div.getElementsByClassName || div.getElementsByClassName("e").length === 0 ) {
		return;
	}

	// Safari caches class attributes, doesn't catch changes (in 3.2)
	div.lastChild.className = "e";

	if ( div.getElementsByClassName("e").length === 1 ) {
		return;
	}
	
	Expr.order.splice(1, 0, "CLASS");
	Expr.find.CLASS = function( match, context, isXML ) {
		if ( typeof context.getElementsByClassName !== "undefined" && !isXML ) {
			return context.getElementsByClassName(match[1]);
		}
	};

	// release memory in IE
	div = null;
})();

function dirNodeCheck( dir, cur, doneName, checkSet, nodeCheck, isXML ) {
	for ( var i = 0, l = checkSet.length; i < l; i++ ) {
		var elem = checkSet[i];

		if ( elem ) {
			var match = false;

			elem = elem[dir];

			while ( elem ) {
				if ( elem.sizcache === doneName ) {
					match = checkSet[elem.sizset];
					break;
				}

				if ( elem.nodeType === 1 && !isXML ){
					elem.sizcache = doneName;
					elem.sizset = i;
				}

				if ( elem.nodeName.toLowerCase() === cur ) {
					match = elem;
					break;
				}

				elem = elem[dir];
			}

			checkSet[i] = match;
		}
	}
}

function dirCheck( dir, cur, doneName, checkSet, nodeCheck, isXML ) {
	for ( var i = 0, l = checkSet.length; i < l; i++ ) {
		var elem = checkSet[i];

		if ( elem ) {
			var match = false;
			
			elem = elem[dir];

			while ( elem ) {
				if ( elem.sizcache === doneName ) {
					match = checkSet[elem.sizset];
					break;
				}

				if ( elem.nodeType === 1 ) {
					if ( !isXML ) {
						elem.sizcache = doneName;
						elem.sizset = i;
					}

					if ( typeof cur !== "string" ) {
						if ( elem === cur ) {
							match = true;
							break;
						}

					} else if ( Sizzle.filter( cur, [elem] ).length > 0 ) {
						match = elem;
						break;
					}
				}

				elem = elem[dir];
			}

			checkSet[i] = match;
		}
	}
}

if ( document.documentElement.contains ) {
	Sizzle.contains = function( a, b ) {
		return a !== b && (a.contains ? a.contains(b) : true);
	};

} else if ( document.documentElement.compareDocumentPosition ) {
	Sizzle.contains = function( a, b ) {
		return !!(a.compareDocumentPosition(b) & 16);
	};

} else {
	Sizzle.contains = function() {
		return false;
	};
}

Sizzle.isXML = function( elem ) {
	// documentElement is verified for cases where it doesn't yet exist
	// (such as loading iframes in IE - #4833) 
	var documentElement = (elem ? elem.ownerDocument || elem : 0).documentElement;

	return documentElement ? documentElement.nodeName !== "HTML" : false;
};

var posProcess = function( selector, context ) {
	var match,
		tmpSet = [],
		later = "",
		root = context.nodeType ? [context] : context;

	// Position selectors must be done after the filter
	// And so must :not(positional) so we move all PSEUDOs to the end
	while ( (match = Expr.match.PSEUDO.exec( selector )) ) {
		later += match[0];
		selector = selector.replace( Expr.match.PSEUDO, "" );
	}

	selector = Expr.relative[selector] ? selector + "*" : selector;

	for ( var i = 0, l = root.length; i < l; i++ ) {
		Sizzle( selector, root[i], tmpSet );
	}

	return Sizzle.filter( later, tmpSet );
};

// EXPOSE
jQuery.find = Sizzle;
jQuery.expr = Sizzle.selectors;
jQuery.expr[":"] = jQuery.expr.filters;
jQuery.unique = Sizzle.uniqueSort;
jQuery.text = Sizzle.getText;
jQuery.isXMLDoc = Sizzle.isXML;
jQuery.contains = Sizzle.contains;


})();


var runtil = /Until$/,
	rparentsprev = /^(?:parents|prevUntil|prevAll)/,
	// Note: This RegExp should be improved, or likely pulled from Sizzle
	rmultiselector = /,/,
	isSimple = /^.[^:#\[\.,]*$/,
	slice = Array.prototype.slice,
	POS = jQuery.expr.match.POS,
	// methods guaranteed to produce a unique set when starting from a unique set
	guaranteedUnique = {
		children: true,
		contents: true,
		next: true,
		prev: true
	};

jQuery.fn.extend({
	find: function( selector ) {
		var self = this,
			i, l;

		if ( typeof selector !== "string" ) {
			return jQuery( selector ).filter(function() {
				for ( i = 0, l = self.length; i < l; i++ ) {
					if ( jQuery.contains( self[ i ], this ) ) {
						return true;
					}
				}
			});
		}

		var ret = this.pushStack( "", "find", selector ),
			length, n, r;

		for ( i = 0, l = this.length; i < l; i++ ) {
			length = ret.length;
			jQuery.find( selector, this[i], ret );

			if ( i > 0 ) {
				// Make sure that the results are unique
				for ( n = length; n < ret.length; n++ ) {
					for ( r = 0; r < length; r++ ) {
						if ( ret[r] === ret[n] ) {
							ret.splice(n--, 1);
							break;
						}
					}
				}
			}
		}

		return ret;
	},

	has: function( target ) {
		var targets = jQuery( target );
		return this.filter(function() {
			for ( var i = 0, l = targets.length; i < l; i++ ) {
				if ( jQuery.contains( this, targets[i] ) ) {
					return true;
				}
			}
		});
	},

	not: function( selector ) {
		return this.pushStack( winnow(this, selector, false), "not", selector);
	},

	filter: function( selector ) {
		return this.pushStack( winnow(this, selector, true), "filter", selector );
	},

	is: function( selector ) {
		return !!selector && ( typeof selector === "string" ?
			jQuery.filter( selector, this ).length > 0 :
			this.filter( selector ).length > 0 );
	},

	closest: function( selectors, context ) {
		var ret = [], i, l, cur = this[0];
		
		// Array
		if ( jQuery.isArray( selectors ) ) {
			var match, selector,
				matches = {},
				level = 1;

			if ( cur && selectors.length ) {
				for ( i = 0, l = selectors.length; i < l; i++ ) {
					selector = selectors[i];

					if ( !matches[ selector ] ) {
						matches[ selector ] = POS.test( selector ) ?
							jQuery( selector, context || this.context ) :
							selector;
					}
				}

				while ( cur && cur.ownerDocument && cur !== context ) {
					for ( selector in matches ) {
						match = matches[ selector ];

						if ( match.jquery ? match.index( cur ) > -1 : jQuery( cur ).is( match ) ) {
							ret.push({ selector: selector, elem: cur, level: level });
						}
					}

					cur = cur.parentNode;
					level++;
				}
			}

			return ret;
		}

		// String
		var pos = POS.test( selectors ) || typeof selectors !== "string" ?
				jQuery( selectors, context || this.context ) :
				0;

		for ( i = 0, l = this.length; i < l; i++ ) {
			cur = this[i];

			while ( cur ) {
				if ( pos ? pos.index(cur) > -1 : jQuery.find.matchesSelector(cur, selectors) ) {
					ret.push( cur );
					break;

				} else {
					cur = cur.parentNode;
					if ( !cur || !cur.ownerDocument || cur === context || cur.nodeType === 11 ) {
						break;
					}
				}
			}
		}

		ret = ret.length > 1 ? jQuery.unique( ret ) : ret;

		return this.pushStack( ret, "closest", selectors );
	},

	// Determine the position of an element within
	// the matched set of elements
	index: function( elem ) {

		// No argument, return index in parent
		if ( !elem ) {
			return ( this[0] && this[0].parentNode ) ? this.prevAll().length : -1;
		}

		// index in selector
		if ( typeof elem === "string" ) {
			return jQuery.inArray( this[0], jQuery( elem ) );
		}

		// Locate the position of the desired element
		return jQuery.inArray(
			// If it receives a jQuery object, the first element is used
			elem.jquery ? elem[0] : elem, this );
	},

	add: function( selector, context ) {
		var set = typeof selector === "string" ?
				jQuery( selector, context ) :
				jQuery.makeArray( selector && selector.nodeType ? [ selector ] : selector ),
			all = jQuery.merge( this.get(), set );

		return this.pushStack( isDisconnected( set[0] ) || isDisconnected( all[0] ) ?
			all :
			jQuery.unique( all ) );
	},

	andSelf: function() {
		return this.add( this.prevObject );
	}
});

// A painfully simple check to see if an element is disconnected
// from a document (should be improved, where feasible).
function isDisconnected( node ) {
	return !node || !node.parentNode || node.parentNode.nodeType === 11;
}

jQuery.each({
	parent: function( elem ) {
		var parent = elem.parentNode;
		return parent && parent.nodeType !== 11 ? parent : null;
	},
	parents: function( elem ) {
		return jQuery.dir( elem, "parentNode" );
	},
	parentsUntil: function( elem, i, until ) {
		return jQuery.dir( elem, "parentNode", until );
	},
	next: function( elem ) {
		return jQuery.nth( elem, 2, "nextSibling" );
	},
	prev: function( elem ) {
		return jQuery.nth( elem, 2, "previousSibling" );
	},
	nextAll: function( elem ) {
		return jQuery.dir( elem, "nextSibling" );
	},
	prevAll: function( elem ) {
		return jQuery.dir( elem, "previousSibling" );
	},
	nextUntil: function( elem, i, until ) {
		return jQuery.dir( elem, "nextSibling", until );
	},
	prevUntil: function( elem, i, until ) {
		return jQuery.dir( elem, "previousSibling", until );
	},
	siblings: function( elem ) {
		return jQuery.sibling( elem.parentNode.firstChild, elem );
	},
	children: function( elem ) {
		return jQuery.sibling( elem.firstChild );
	},
	contents: function( elem ) {
		return jQuery.nodeName( elem, "iframe" ) ?
			elem.contentDocument || elem.contentWindow.document :
			jQuery.makeArray( elem.childNodes );
	}
}, function( name, fn ) {
	jQuery.fn[ name ] = function( until, selector ) {
		var ret = jQuery.map( this, fn, until ),
			// The variable 'args' was introduced in
			// https://github.com/jquery/jquery/commit/52a0238
			// to work around a bug in Chrome 10 (Dev) and should be removed when the bug is fixed.
			// http://code.google.com/p/v8/issues/detail?id=1050
			args = slice.call(arguments);

		if ( !runtil.test( name ) ) {
			selector = until;
		}

		if ( selector && typeof selector === "string" ) {
			ret = jQuery.filter( selector, ret );
		}

		ret = this.length > 1 && !guaranteedUnique[ name ] ? jQuery.unique( ret ) : ret;

		if ( (this.length > 1 || rmultiselector.test( selector )) && rparentsprev.test( name ) ) {
			ret = ret.reverse();
		}

		return this.pushStack( ret, name, args.join(",") );
	};
});

jQuery.extend({
	filter: function( expr, elems, not ) {
		if ( not ) {
			expr = ":not(" + expr + ")";
		}

		return elems.length === 1 ?
			jQuery.find.matchesSelector(elems[0], expr) ? [ elems[0] ] : [] :
			jQuery.find.matches(expr, elems);
	},

	dir: function( elem, dir, until ) {
		var matched = [],
			cur = elem[ dir ];

		while ( cur && cur.nodeType !== 9 && (until === undefined || cur.nodeType !== 1 || !jQuery( cur ).is( until )) ) {
			if ( cur.nodeType === 1 ) {
				matched.push( cur );
			}
			cur = cur[dir];
		}
		return matched;
	},

	nth: function( cur, result, dir, elem ) {
		result = result || 1;
		var num = 0;

		for ( ; cur; cur = cur[dir] ) {
			if ( cur.nodeType === 1 && ++num === result ) {
				break;
			}
		}

		return cur;
	},

	sibling: function( n, elem ) {
		var r = [];

		for ( ; n; n = n.nextSibling ) {
			if ( n.nodeType === 1 && n !== elem ) {
				r.push( n );
			}
		}

		return r;
	}
});

// Implement the identical functionality for filter and not
function winnow( elements, qualifier, keep ) {

	// Can't pass null or undefined to indexOf in Firefox 4
	// Set to 0 to skip string check
	qualifier = qualifier || 0;

	if ( jQuery.isFunction( qualifier ) ) {
		return jQuery.grep(elements, function( elem, i ) {
			var retVal = !!qualifier.call( elem, i, elem );
			return retVal === keep;
		});

	} else if ( qualifier.nodeType ) {
		return jQuery.grep(elements, function( elem, i ) {
			return (elem === qualifier) === keep;
		});

	} else if ( typeof qualifier === "string" ) {
		var filtered = jQuery.grep(elements, function( elem ) {
			return elem.nodeType === 1;
		});

		if ( isSimple.test( qualifier ) ) {
			return jQuery.filter(qualifier, filtered, !keep);
		} else {
			qualifier = jQuery.filter( qualifier, filtered );
		}
	}

	return jQuery.grep(elements, function( elem, i ) {
		return (jQuery.inArray( elem, qualifier ) >= 0) === keep;
	});
}




var rinlinejQuery = / jQuery\d+="(?:\d+|null)"/g,
	rleadingWhitespace = /^\s+/,
	rxhtmlTag = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\w:]+)[^>]*)\/>/ig,
	rtagName = /<([\w:]+)/,
	rtbody = /<tbody/i,
	rhtml = /<|&#?\w+;/,
	rnocache = /<(?:script|object|embed|option|style)/i,
	// checked="checked" or checked
	rchecked = /checked\s*(?:[^=]|=\s*.checked.)/i,
	rscriptType = /\/(java|ecma)script/i,
	rcleanScript = /^\s*<!(?:\[CDATA\[|\-\-)/,
	wrapMap = {
		option: [ 1, "<select multiple='multiple'>", "</select>" ],
		legend: [ 1, "<fieldset>", "</fieldset>" ],
		thead: [ 1, "<table>", "</table>" ],
		tr: [ 2, "<table><tbody>", "</tbody></table>" ],
		td: [ 3, "<table><tbody><tr>", "</tr></tbody></table>" ],
		col: [ 2, "<table><tbody></tbody><colgroup>", "</colgroup></table>" ],
		area: [ 1, "<map>", "</map>" ],
		_default: [ 0, "", "" ]
	};

wrapMap.optgroup = wrapMap.option;
wrapMap.tbody = wrapMap.tfoot = wrapMap.colgroup = wrapMap.caption = wrapMap.thead;
wrapMap.th = wrapMap.td;

// IE can't serialize <link> and <script> tags normally
if ( !jQuery.support.htmlSerialize ) {
	wrapMap._default = [ 1, "div<div>", "</div>" ];
}

jQuery.fn.extend({
	text: function( text ) {
		if ( jQuery.isFunction(text) ) {
			return this.each(function(i) {
				var self = jQuery( this );

				self.text( text.call(this, i, self.text()) );
			});
		}

		if ( typeof text !== "object" && text !== undefined ) {
			return this.empty().append( (this[0] && this[0].ownerDocument || document).createTextNode( text ) );
		}

		return jQuery.text( this );
	},

	wrapAll: function( html ) {
		if ( jQuery.isFunction( html ) ) {
			return this.each(function(i) {
				jQuery(this).wrapAll( html.call(this, i) );
			});
		}

		if ( this[0] ) {
			// The elements to wrap the target around
			var wrap = jQuery( html, this[0].ownerDocument ).eq(0).clone(true);

			if ( this[0].parentNode ) {
				wrap.insertBefore( this[0] );
			}

			wrap.map(function() {
				var elem = this;

				while ( elem.firstChild && elem.firstChild.nodeType === 1 ) {
					elem = elem.firstChild;
				}

				return elem;
			}).append( this );
		}

		return this;
	},

	wrapInner: function( html ) {
		if ( jQuery.isFunction( html ) ) {
			return this.each(function(i) {
				jQuery(this).wrapInner( html.call(this, i) );
			});
		}

		return this.each(function() {
			var self = jQuery( this ),
				contents = self.contents();

			if ( contents.length ) {
				contents.wrapAll( html );

			} else {
				self.append( html );
			}
		});
	},

	wrap: function( html ) {
		return this.each(function() {
			jQuery( this ).wrapAll( html );
		});
	},

	unwrap: function() {
		return this.parent().each(function() {
			if ( !jQuery.nodeName( this, "body" ) ) {
				jQuery( this ).replaceWith( this.childNodes );
			}
		}).end();
	},

	append: function() {
		return this.domManip(arguments, true, function( elem ) {
			if ( this.nodeType === 1 ) {
				this.appendChild( elem );
			}
		});
	},

	prepend: function() {
		return this.domManip(arguments, true, function( elem ) {
			if ( this.nodeType === 1 ) {
				this.insertBefore( elem, this.firstChild );
			}
		});
	},

	before: function() {
		if ( this[0] && this[0].parentNode ) {
			return this.domManip(arguments, false, function( elem ) {
				this.parentNode.insertBefore( elem, this );
			});
		} else if ( arguments.length ) {
			var set = jQuery(arguments[0]);
			set.push.apply( set, this.toArray() );
			return this.pushStack( set, "before", arguments );
		}
	},

	after: function() {
		if ( this[0] && this[0].parentNode ) {
			return this.domManip(arguments, false, function( elem ) {
				this.parentNode.insertBefore( elem, this.nextSibling );
			});
		} else if ( arguments.length ) {
			var set = this.pushStack( this, "after", arguments );
			set.push.apply( set, jQuery(arguments[0]).toArray() );
			return set;
		}
	},

	// keepData is for internal use only--do not document
	remove: function( selector, keepData ) {
		for ( var i = 0, elem; (elem = this[i]) != null; i++ ) {
			if ( !selector || jQuery.filter( selector, [ elem ] ).length ) {
				if ( !keepData && elem.nodeType === 1 ) {
					jQuery.cleanData( elem.getElementsByTagName("*") );
					jQuery.cleanData( [ elem ] );
				}

				if ( elem.parentNode ) {
					elem.parentNode.removeChild( elem );
				}
			}
		}

		return this;
	},

	empty: function() {
		for ( var i = 0, elem; (elem = this[i]) != null; i++ ) {
			// Remove element nodes and prevent memory leaks
			if ( elem.nodeType === 1 ) {
				jQuery.cleanData( elem.getElementsByTagName("*") );
			}

			// Remove any remaining nodes
			while ( elem.firstChild ) {
				elem.removeChild( elem.firstChild );
			}
		}

		return this;
	},

	clone: function( dataAndEvents, deepDataAndEvents ) {
		dataAndEvents = dataAndEvents == null ? false : dataAndEvents;
		deepDataAndEvents = deepDataAndEvents == null ? dataAndEvents : deepDataAndEvents;

		return this.map( function () {
			return jQuery.clone( this, dataAndEvents, deepDataAndEvents );
		});
	},

	html: function( value ) {
		if ( value === undefined ) {
			return this[0] && this[0].nodeType === 1 ?
				this[0].innerHTML.replace(rinlinejQuery, "") :
				null;

		// See if we can take a shortcut and just use innerHTML
		} else if ( typeof value === "string" && !rnocache.test( value ) &&
			(jQuery.support.leadingWhitespace || !rleadingWhitespace.test( value )) &&
			!wrapMap[ (rtagName.exec( value ) || ["", ""])[1].toLowerCase() ] ) {

			value = value.replace(rxhtmlTag, "<$1></$2>");

			try {
				for ( var i = 0, l = this.length; i < l; i++ ) {
					// Remove element nodes and prevent memory leaks
					if ( this[i].nodeType === 1 ) {
						jQuery.cleanData( this[i].getElementsByTagName("*") );
						this[i].innerHTML = value;
					}
				}

			// If using innerHTML throws an exception, use the fallback method
			} catch(e) {
				this.empty().append( value );
			}

		} else if ( jQuery.isFunction( value ) ) {
			this.each(function(i){
				var self = jQuery( this );

				self.html( value.call(this, i, self.html()) );
			});

		} else {
			this.empty().append( value );
		}

		return this;
	},

	replaceWith: function( value ) {
		if ( this[0] && this[0].parentNode ) {
			// Make sure that the elements are removed from the DOM before they are inserted
			// this can help fix replacing a parent with child elements
			if ( jQuery.isFunction( value ) ) {
				return this.each(function(i) {
					var self = jQuery(this), old = self.html();
					self.replaceWith( value.call( this, i, old ) );
				});
			}

			if ( typeof value !== "string" ) {
				value = jQuery( value ).detach();
			}

			return this.each(function() {
				var next = this.nextSibling,
					parent = this.parentNode;

				jQuery( this ).remove();

				if ( next ) {
					jQuery(next).before( value );
				} else {
					jQuery(parent).append( value );
				}
			});
		} else {
			return this.length ?
				this.pushStack( jQuery(jQuery.isFunction(value) ? value() : value), "replaceWith", value ) :
				this;
		}
	},

	detach: function( selector ) {
		return this.remove( selector, true );
	},

	domManip: function( args, table, callback ) {
		var results, first, fragment, parent,
			value = args[0],
			scripts = [];

		// We can't cloneNode fragments that contain checked, in WebKit
		if ( !jQuery.support.checkClone && arguments.length === 3 && typeof value === "string" && rchecked.test( value ) ) {
			return this.each(function() {
				jQuery(this).domManip( args, table, callback, true );
			});
		}

		if ( jQuery.isFunction(value) ) {
			return this.each(function(i) {
				var self = jQuery(this);
				args[0] = value.call(this, i, table ? self.html() : undefined);
				self.domManip( args, table, callback );
			});
		}

		if ( this[0] ) {
			parent = value && value.parentNode;

			// If we're in a fragment, just use that instead of building a new one
			if ( jQuery.support.parentNode && parent && parent.nodeType === 11 && parent.childNodes.length === this.length ) {
				results = { fragment: parent };

			} else {
				results = jQuery.buildFragment( args, this, scripts );
			}

			fragment = results.fragment;

			if ( fragment.childNodes.length === 1 ) {
				first = fragment = fragment.firstChild;
			} else {
				first = fragment.firstChild;
			}

			if ( first ) {
				table = table && jQuery.nodeName( first, "tr" );

				for ( var i = 0, l = this.length, lastIndex = l - 1; i < l; i++ ) {
					callback.call(
						table ?
							root(this[i], first) :
							this[i],
						// Make sure that we do not leak memory by inadvertently discarding
						// the original fragment (which might have attached data) instead of
						// using it; in addition, use the original fragment object for the last
						// item instead of first because it can end up being emptied incorrectly
						// in certain situations (Bug #8070).
						// Fragments from the fragment cache must always be cloned and never used
						// in place.
						results.cacheable || (l > 1 && i < lastIndex) ?
							jQuery.clone( fragment, true, true ) :
							fragment
					);
				}
			}

			if ( scripts.length ) {
				jQuery.each( scripts, evalScript );
			}
		}

		return this;
	}
});

function root( elem, cur ) {
	return jQuery.nodeName(elem, "table") ?
		(elem.getElementsByTagName("tbody")[0] ||
		elem.appendChild(elem.ownerDocument.createElement("tbody"))) :
		elem;
}

function cloneCopyEvent( src, dest ) {

	if ( dest.nodeType !== 1 || !jQuery.hasData( src ) ) {
		return;
	}

	var internalKey = jQuery.expando,
		oldData = jQuery.data( src ),
		curData = jQuery.data( dest, oldData );

	// Switch to use the internal data object, if it exists, for the next
	// stage of data copying
	if ( (oldData = oldData[ internalKey ]) ) {
		var events = oldData.events;
				curData = curData[ internalKey ] = jQuery.extend({}, oldData);

		if ( events ) {
			delete curData.handle;
			curData.events = {};

			for ( var type in events ) {
				for ( var i = 0, l = events[ type ].length; i < l; i++ ) {
					jQuery.event.add( dest, type + ( events[ type ][ i ].namespace ? "." : "" ) + events[ type ][ i ].namespace, events[ type ][ i ], events[ type ][ i ].data );
				}
			}
		}
	}
}

function cloneFixAttributes( src, dest ) {
	var nodeName;

	// We do not need to do anything for non-Elements
	if ( dest.nodeType !== 1 ) {
		return;
	}

	// clearAttributes removes the attributes, which we don't want,
	// but also removes the attachEvent events, which we *do* want
	if ( dest.clearAttributes ) {
		dest.clearAttributes();
	}

	// mergeAttributes, in contrast, only merges back on the
	// original attributes, not the events
	if ( dest.mergeAttributes ) {
		dest.mergeAttributes( src );
	}

	nodeName = dest.nodeName.toLowerCase();

	// IE6-8 fail to clone children inside object elements that use
	// the proprietary classid attribute value (rather than the type
	// attribute) to identify the type of content to display
	if ( nodeName === "object" ) {
		dest.outerHTML = src.outerHTML;

	} else if ( nodeName === "input" && (src.type === "checkbox" || src.type === "radio") ) {
		// IE6-8 fails to persist the checked state of a cloned checkbox
		// or radio button. Worse, IE6-7 fail to give the cloned element
		// a checked appearance if the defaultChecked value isn't also set
		if ( src.checked ) {
			dest.defaultChecked = dest.checked = src.checked;
		}

		// IE6-7 get confused and end up setting the value of a cloned
		// checkbox/radio button to an empty string instead of "on"
		if ( dest.value !== src.value ) {
			dest.value = src.value;
		}

	// IE6-8 fails to return the selected option to the default selected
	// state when cloning options
	} else if ( nodeName === "option" ) {
		dest.selected = src.defaultSelected;

	// IE6-8 fails to set the defaultValue to the correct value when
	// cloning other types of input fields
	} else if ( nodeName === "input" || nodeName === "textarea" ) {
		dest.defaultValue = src.defaultValue;
	}

	// Event data gets referenced instead of copied if the expando
	// gets copied too
	dest.removeAttribute( jQuery.expando );
}

jQuery.buildFragment = function( args, nodes, scripts ) {
	var fragment, cacheable, cacheresults, doc;

  // nodes may contain either an explicit document object,
  // a jQuery collection or context object.
  // If nodes[0] contains a valid object to assign to doc
  if ( nodes && nodes[0] ) {
    doc = nodes[0].ownerDocument || nodes[0];
  }

  // Ensure that an attr object doesn't incorrectly stand in as a document object
	// Chrome and Firefox seem to allow this to occur and will throw exception
	// Fixes #8950
	if ( !doc.createDocumentFragment ) {
		doc = document;
	}

	// Only cache "small" (1/2 KB) HTML strings that are associated with the main document
	// Cloning options loses the selected state, so don't cache them
	// IE 6 doesn't like it when you put <object> or <embed> elements in a fragment
	// Also, WebKit does not clone 'checked' attributes on cloneNode, so don't cache
	if ( args.length === 1 && typeof args[0] === "string" && args[0].length < 512 && doc === document &&
		args[0].charAt(0) === "<" && !rnocache.test( args[0] ) && (jQuery.support.checkClone || !rchecked.test( args[0] )) ) {

		cacheable = true;

		cacheresults = jQuery.fragments[ args[0] ];
		if ( cacheresults && cacheresults !== 1 ) {
			fragment = cacheresults;
		}
	}

	if ( !fragment ) {
		fragment = doc.createDocumentFragment();
		jQuery.clean( args, doc, fragment, scripts );
	}

	if ( cacheable ) {
		jQuery.fragments[ args[0] ] = cacheresults ? fragment : 1;
	}

	return { fragment: fragment, cacheable: cacheable };
};

jQuery.fragments = {};

jQuery.each({
	appendTo: "append",
	prependTo: "prepend",
	insertBefore: "before",
	insertAfter: "after",
	replaceAll: "replaceWith"
}, function( name, original ) {
	jQuery.fn[ name ] = function( selector ) {
		var ret = [],
			insert = jQuery( selector ),
			parent = this.length === 1 && this[0].parentNode;

		if ( parent && parent.nodeType === 11 && parent.childNodes.length === 1 && insert.length === 1 ) {
			insert[ original ]( this[0] );
			return this;

		} else {
			for ( var i = 0, l = insert.length; i < l; i++ ) {
				var elems = (i > 0 ? this.clone(true) : this).get();
				jQuery( insert[i] )[ original ]( elems );
				ret = ret.concat( elems );
			}

			return this.pushStack( ret, name, insert.selector );
		}
	};
});

function getAll( elem ) {
	if ( "getElementsByTagName" in elem ) {
		return elem.getElementsByTagName( "*" );

	} else if ( "querySelectorAll" in elem ) {
		return elem.querySelectorAll( "*" );

	} else {
		return [];
	}
}

// Used in clean, fixes the defaultChecked property
function fixDefaultChecked( elem ) {
	if ( elem.type === "checkbox" || elem.type === "radio" ) {
		elem.defaultChecked = elem.checked;
	}
}
// Finds all inputs and passes them to fixDefaultChecked
function findInputs( elem ) {
	if ( jQuery.nodeName( elem, "input" ) ) {
		fixDefaultChecked( elem );
	} else if ( "getElementsByTagName" in elem ) {
		jQuery.grep( elem.getElementsByTagName("input"), fixDefaultChecked );
	}
}

jQuery.extend({
	clone: function( elem, dataAndEvents, deepDataAndEvents ) {
		var clone = elem.cloneNode(true),
				srcElements,
				destElements,
				i;

		if ( (!jQuery.support.noCloneEvent || !jQuery.support.noCloneChecked) &&
				(elem.nodeType === 1 || elem.nodeType === 11) && !jQuery.isXMLDoc(elem) ) {
			// IE copies events bound via attachEvent when using cloneNode.
			// Calling detachEvent on the clone will also remove the events
			// from the original. In order to get around this, we use some
			// proprietary methods to clear the events. Thanks to MooTools
			// guys for this hotness.

			cloneFixAttributes( elem, clone );

			// Using Sizzle here is crazy slow, so we use getElementsByTagName
			// instead
			srcElements = getAll( elem );
			destElements = getAll( clone );

			// Weird iteration because IE will replace the length property
			// with an element if you are cloning the body and one of the
			// elements on the page has a name or id of "length"
			for ( i = 0; srcElements[i]; ++i ) {
				// Ensure that the destination node is not null; Fixes #9587
				if ( destElements[i] ) {
					cloneFixAttributes( srcElements[i], destElements[i] );
				}
			}
		}

		// Copy the events from the original to the clone
		if ( dataAndEvents ) {
			cloneCopyEvent( elem, clone );

			if ( deepDataAndEvents ) {
				srcElements = getAll( elem );
				destElements = getAll( clone );

				for ( i = 0; srcElements[i]; ++i ) {
					cloneCopyEvent( srcElements[i], destElements[i] );
				}
			}
		}

		srcElements = destElements = null;

		// Return the cloned set
		return clone;
	},

	clean: function( elems, context, fragment, scripts ) {
		var checkScriptType;

		context = context || document;

		// !context.createElement fails in IE with an error but returns typeof 'object'
		if ( typeof context.createElement === "undefined" ) {
			context = context.ownerDocument || context[0] && context[0].ownerDocument || document;
		}

		var ret = [], j;

		for ( var i = 0, elem; (elem = elems[i]) != null; i++ ) {
			if ( typeof elem === "number" ) {
				elem += "";
			}

			if ( !elem ) {
				continue;
			}

			// Convert html string into DOM nodes
			if ( typeof elem === "string" ) {
				if ( !rhtml.test( elem ) ) {
					elem = context.createTextNode( elem );
				} else {
					// Fix "XHTML"-style tags in all browsers
					elem = elem.replace(rxhtmlTag, "<$1></$2>");

					// Trim whitespace, otherwise indexOf won't work as expected
					var tag = (rtagName.exec( elem ) || ["", ""])[1].toLowerCase(),
						wrap = wrapMap[ tag ] || wrapMap._default,
						depth = wrap[0],
						div = context.createElement("div");

					// Go to html and back, then peel off extra wrappers
					div.innerHTML = wrap[1] + elem + wrap[2];

					// Move to the right depth
					while ( depth-- ) {
						div = div.lastChild;
					}

					// Remove IE's autoinserted <tbody> from table fragments
					if ( !jQuery.support.tbody ) {

						// String was a <table>, *may* have spurious <tbody>
						var hasBody = rtbody.test(elem),
							tbody = tag === "table" && !hasBody ?
								div.firstChild && div.firstChild.childNodes :

								// String was a bare <thead> or <tfoot>
								wrap[1] === "<table>" && !hasBody ?
									div.childNodes :
									[];

						for ( j = tbody.length - 1; j >= 0 ; --j ) {
							if ( jQuery.nodeName( tbody[ j ], "tbody" ) && !tbody[ j ].childNodes.length ) {
								tbody[ j ].parentNode.removeChild( tbody[ j ] );
							}
						}
					}

					// IE completely kills leading whitespace when innerHTML is used
					if ( !jQuery.support.leadingWhitespace && rleadingWhitespace.test( elem ) ) {
						div.insertBefore( context.createTextNode( rleadingWhitespace.exec(elem)[0] ), div.firstChild );
					}

					elem = div.childNodes;
				}
			}

			// Resets defaultChecked for any radios and checkboxes
			// about to be appended to the DOM in IE 6/7 (#8060)
			var len;
			if ( !jQuery.support.appendChecked ) {
				if ( elem[0] && typeof (len = elem.length) === "number" ) {
					for ( j = 0; j < len; j++ ) {
						findInputs( elem[j] );
					}
				} else {
					findInputs( elem );
				}
			}

			if ( elem.nodeType ) {
				ret.push( elem );
			} else {
				ret = jQuery.merge( ret, elem );
			}
		}

		if ( fragment ) {
			checkScriptType = function( elem ) {
				return !elem.type || rscriptType.test( elem.type );
			};
			for ( i = 0; ret[i]; i++ ) {
				if ( scripts && jQuery.nodeName( ret[i], "script" ) && (!ret[i].type || ret[i].type.toLowerCase() === "text/javascript") ) {
					scripts.push( ret[i].parentNode ? ret[i].parentNode.removeChild( ret[i] ) : ret[i] );

				} else {
					if ( ret[i].nodeType === 1 ) {
						var jsTags = jQuery.grep( ret[i].getElementsByTagName( "script" ), checkScriptType );

						ret.splice.apply( ret, [i + 1, 0].concat( jsTags ) );
					}
					fragment.appendChild( ret[i] );
				}
			}
		}

		return ret;
	},

	cleanData: function( elems ) {
		var data, id, cache = jQuery.cache, internalKey = jQuery.expando, special = jQuery.event.special,
			deleteExpando = jQuery.support.deleteExpando;

		for ( var i = 0, elem; (elem = elems[i]) != null; i++ ) {
			if ( elem.nodeName && jQuery.noData[elem.nodeName.toLowerCase()] ) {
				continue;
			}

			id = elem[ jQuery.expando ];

			if ( id ) {
				data = cache[ id ] && cache[ id ][ internalKey ];

				if ( data && data.events ) {
					for ( var type in data.events ) {
						if ( special[ type ] ) {
							jQuery.event.remove( elem, type );

						// This is a shortcut to avoid jQuery.event.remove's overhead
						} else {
							jQuery.removeEvent( elem, type, data.handle );
						}
					}

					// Null the DOM reference to avoid IE6/7/8 leak (#7054)
					if ( data.handle ) {
						data.handle.elem = null;
					}
				}

				if ( deleteExpando ) {
					delete elem[ jQuery.expando ];

				} else if ( elem.removeAttribute ) {
					elem.removeAttribute( jQuery.expando );
				}

				delete cache[ id ];
			}
		}
	}
});

function evalScript( i, elem ) {
	if ( elem.src ) {
		jQuery.ajax({
			url: elem.src,
			async: false,
			dataType: "script"
		});
	} else {
		jQuery.globalEval( ( elem.text || elem.textContent || elem.innerHTML || "" ).replace( rcleanScript, "/*$0*/" ) );
	}

	if ( elem.parentNode ) {
		elem.parentNode.removeChild( elem );
	}
}




var ralpha = /alpha\([^)]*\)/i,
	ropacity = /opacity=([^)]*)/,
	// fixed for IE9, see #8346
	rupper = /([A-Z]|^ms)/g,
	rnumpx = /^-?\d+(?:px)?$/i,
	rnum = /^-?\d/,
	rrelNum = /^([\-+])=([\-+.\de]+)/,

	cssShow = { position: "absolute", visibility: "hidden", display: "block" },
	cssWidth = [ "Left", "Right" ],
	cssHeight = [ "Top", "Bottom" ],
	curCSS,

	getComputedStyle,
	currentStyle;

jQuery.fn.css = function( name, value ) {
	// Setting 'undefined' is a no-op
	if ( arguments.length === 2 && value === undefined ) {
		return this;
	}

	return jQuery.access( this, name, value, true, function( elem, name, value ) {
		return value !== undefined ?
			jQuery.style( elem, name, value ) :
			jQuery.css( elem, name );
	});
};

jQuery.extend({
	// Add in style property hooks for overriding the default
	// behavior of getting and setting a style property
	cssHooks: {
		opacity: {
			get: function( elem, computed ) {
				if ( computed ) {
					// We should always get a number back from opacity
					var ret = curCSS( elem, "opacity", "opacity" );
					return ret === "" ? "1" : ret;

				} else {
					return elem.style.opacity;
				}
			}
		}
	},

	// Exclude the following css properties to add px
	cssNumber: {
		"fillOpacity": true,
		"fontWeight": true,
		"lineHeight": true,
		"opacity": true,
		"orphans": true,
		"widows": true,
		"zIndex": true,
		"zoom": true
	},

	// Add in properties whose names you wish to fix before
	// setting or getting the value
	cssProps: {
		// normalize float css property
		"float": jQuery.support.cssFloat ? "cssFloat" : "styleFloat"
	},

	// Get and set the style property on a DOM Node
	style: function( elem, name, value, extra ) {
		// Don't set styles on text and comment nodes
		if ( !elem || elem.nodeType === 3 || elem.nodeType === 8 || !elem.style ) {
			return;
		}

		// Make sure that we're working with the right name
		var ret, type, origName = jQuery.camelCase( name ),
			style = elem.style, hooks = jQuery.cssHooks[ origName ];

		name = jQuery.cssProps[ origName ] || origName;

		// Check if we're setting a value
		if ( value !== undefined ) {
			type = typeof value;

			// convert relative number strings (+= or -=) to relative numbers. #7345
			if ( type === "string" && (ret = rrelNum.exec( value )) ) {
				value = ( +( ret[1] + 1) * +ret[2] ) + parseFloat( jQuery.css( elem, name ) );
				// Fixes bug #9237
				type = "number";
			}

			// Make sure that NaN and null values aren't set. See: #7116
			if ( value == null || type === "number" && isNaN( value ) ) {
				return;
			}

			// If a number was passed in, add 'px' to the (except for certain CSS properties)
			if ( type === "number" && !jQuery.cssNumber[ origName ] ) {
				value += "px";
			}

			// If a hook was provided, use that value, otherwise just set the specified value
			if ( !hooks || !("set" in hooks) || (value = hooks.set( elem, value )) !== undefined ) {
				// Wrapped to prevent IE from throwing errors when 'invalid' values are provided
				// Fixes bug #5509
				try {
					style[ name ] = value;
				} catch(e) {}
			}

		} else {
			// If a hook was provided get the non-computed value from there
			if ( hooks && "get" in hooks && (ret = hooks.get( elem, false, extra )) !== undefined ) {
				return ret;
			}

			// Otherwise just get the value from the style object
			return style[ name ];
		}
	},

	css: function( elem, name, extra ) {
		var ret, hooks;

		// Make sure that we're working with the right name
		name = jQuery.camelCase( name );
		hooks = jQuery.cssHooks[ name ];
		name = jQuery.cssProps[ name ] || name;

		// cssFloat needs a special treatment
		if ( name === "cssFloat" ) {
			name = "float";
		}

		// If a hook was provided get the computed value from there
		if ( hooks && "get" in hooks && (ret = hooks.get( elem, true, extra )) !== undefined ) {
			return ret;

		// Otherwise, if a way to get the computed value exists, use that
		} else if ( curCSS ) {
			return curCSS( elem, name );
		}
	},

	// A method for quickly swapping in/out CSS properties to get correct calculations
	swap: function( elem, options, callback ) {
		var old = {};

		// Remember the old values, and insert the new ones
		for ( var name in options ) {
			old[ name ] = elem.style[ name ];
			elem.style[ name ] = options[ name ];
		}

		callback.call( elem );

		// Revert the old values
		for ( name in options ) {
			elem.style[ name ] = old[ name ];
		}
	}
});

// DEPRECATED, Use jQuery.css() instead
jQuery.curCSS = jQuery.css;

jQuery.each(["height", "width"], function( i, name ) {
	jQuery.cssHooks[ name ] = {
		get: function( elem, computed, extra ) {
			var val;

			if ( computed ) {
				if ( elem.offsetWidth !== 0 ) {
					return getWH( elem, name, extra );
				} else {
					jQuery.swap( elem, cssShow, function() {
						val = getWH( elem, name, extra );
					});
				}

				return val;
			}
		},

		set: function( elem, value ) {
			if ( rnumpx.test( value ) ) {
				// ignore negative width and height values #1599
				value = parseFloat( value );

				if ( value >= 0 ) {
					return value + "px";
				}

			} else {
				return value;
			}
		}
	};
});

if ( !jQuery.support.opacity ) {
	jQuery.cssHooks.opacity = {
		get: function( elem, computed ) {
			// IE uses filters for opacity
			return ropacity.test( (computed && elem.currentStyle ? elem.currentStyle.filter : elem.style.filter) || "" ) ?
				( parseFloat( RegExp.$1 ) / 100 ) + "" :
				computed ? "1" : "";
		},

		set: function( elem, value ) {
			var style = elem.style,
				currentStyle = elem.currentStyle,
				opacity = jQuery.isNaN( value ) ? "" : "alpha(opacity=" + value * 100 + ")",
				filter = currentStyle && currentStyle.filter || style.filter || "";

			// IE has trouble with opacity if it does not have layout
			// Force it by setting the zoom level
			style.zoom = 1;

			// if setting opacity to 1, and no other filters exist - attempt to remove filter attribute #6652
			if ( value >= 1 && jQuery.trim( filter.replace( ralpha, "" ) ) === "" ) {

				// Setting style.filter to null, "" & " " still leave "filter:" in the cssText
				// if "filter:" is present at all, clearType is disabled, we want to avoid this
				// style.removeAttribute is IE Only, but so apparently is this code path...
				style.removeAttribute( "filter" );

				// if there there is no filter style applied in a css rule, we are done
				if ( currentStyle && !currentStyle.filter ) {
					return;
				}
			}

			// otherwise, set new filter values
			style.filter = ralpha.test( filter ) ?
				filter.replace( ralpha, opacity ) :
				filter + " " + opacity;
		}
	};
}

jQuery(function() {
	// This hook cannot be added until DOM ready because the support test
	// for it is not run until after DOM ready
	if ( !jQuery.support.reliableMarginRight ) {
		jQuery.cssHooks.marginRight = {
			get: function( elem, computed ) {
				// WebKit Bug 13343 - getComputedStyle returns wrong value for margin-right
				// Work around by temporarily setting element display to inline-block
				var ret;
				jQuery.swap( elem, { "display": "inline-block" }, function() {
					if ( computed ) {
						ret = curCSS( elem, "margin-right", "marginRight" );
					} else {
						ret = elem.style.marginRight;
					}
				});
				return ret;
			}
		};
	}
});

if ( document.defaultView && document.defaultView.getComputedStyle ) {
	getComputedStyle = function( elem, name ) {
		var ret, defaultView, computedStyle;

		name = name.replace( rupper, "-$1" ).toLowerCase();

		if ( !(defaultView = elem.ownerDocument.defaultView) ) {
			return undefined;
		}

		if ( (computedStyle = defaultView.getComputedStyle( elem, null )) ) {
			ret = computedStyle.getPropertyValue( name );
			if ( ret === "" && !jQuery.contains( elem.ownerDocument.documentElement, elem ) ) {
				ret = jQuery.style( elem, name );
			}
		}

		return ret;
	};
}

if ( document.documentElement.currentStyle ) {
	currentStyle = function( elem, name ) {
		var left,
			ret = elem.currentStyle && elem.currentStyle[ name ],
			rsLeft = elem.runtimeStyle && elem.runtimeStyle[ name ],
			style = elem.style;

		// From the awesome hack by Dean Edwards
		// http://erik.eae.net/archives/2007/07/27/18.54.15/#comment-102291

		// If we're not dealing with a regular pixel number
		// but a number that has a weird ending, we need to convert it to pixels
		if ( !rnumpx.test( ret ) && rnum.test( ret ) ) {
			// Remember the original values
			left = style.left;

			// Put in the new values to get a computed value out
			if ( rsLeft ) {
				elem.runtimeStyle.left = elem.currentStyle.left;
			}
			style.left = name === "fontSize" ? "1em" : (ret || 0);
			ret = style.pixelLeft + "px";

			// Revert the changed values
			style.left = left;
			if ( rsLeft ) {
				elem.runtimeStyle.left = rsLeft;
			}
		}

		return ret === "" ? "auto" : ret;
	};
}

curCSS = getComputedStyle || currentStyle;

function getWH( elem, name, extra ) {

	// Start with offset property
	var val = name === "width" ? elem.offsetWidth : elem.offsetHeight,
		which = name === "width" ? cssWidth : cssHeight;

	if ( val > 0 ) {
		if ( extra !== "border" ) {
			jQuery.each( which, function() {
				if ( !extra ) {
					val -= parseFloat( jQuery.css( elem, "padding" + this ) ) || 0;
				}
				if ( extra === "margin" ) {
					val += parseFloat( jQuery.css( elem, extra + this ) ) || 0;
				} else {
					val -= parseFloat( jQuery.css( elem, "border" + this + "Width" ) ) || 0;
				}
			});
		}

		return val + "px";
	}

	// Fall back to computed then uncomputed css if necessary
	val = curCSS( elem, name, name );
	if ( val < 0 || val == null ) {
		val = elem.style[ name ] || 0;
	}
	// Normalize "", auto, and prepare for extra
	val = parseFloat( val ) || 0;

	// Add padding, border, margin
	if ( extra ) {
		jQuery.each( which, function() {
			val += parseFloat( jQuery.css( elem, "padding" + this ) ) || 0;
			if ( extra !== "padding" ) {
				val += parseFloat( jQuery.css( elem, "border" + this + "Width" ) ) || 0;
			}
			if ( extra === "margin" ) {
				val += parseFloat( jQuery.css( elem, extra + this ) ) || 0;
			}
		});
	}

	return val + "px";
}

if ( jQuery.expr && jQuery.expr.filters ) {
	jQuery.expr.filters.hidden = function( elem ) {
		var width = elem.offsetWidth,
			height = elem.offsetHeight;

		return (width === 0 && height === 0) || (!jQuery.support.reliableHiddenOffsets && (elem.style.display || jQuery.css( elem, "display" )) === "none");
	};

	jQuery.expr.filters.visible = function( elem ) {
		return !jQuery.expr.filters.hidden( elem );
	};
}




var r20 = /%20/g,
	rbracket = /\[\]$/,
	rCRLF = /\r?\n/g,
	rhash = /#.*$/,
	rheaders = /^(.*?):[ \t]*([^\r\n]*)\r?$/mg, // IE leaves an \r character at EOL
	rinput = /^(?:color|date|datetime|datetime-local|email|hidden|month|number|password|range|search|tel|text|time|url|week)$/i,
	// #7653, #8125, #8152: local protocol detection
	rlocalProtocol = /^(?:about|app|app\-storage|.+\-extension|file|res|widget):$/,
	rnoContent = /^(?:GET|HEAD)$/,
	rprotocol = /^\/\//,
	rquery = /\?/,
	rscript = /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
	rselectTextarea = /^(?:select|textarea)/i,
	rspacesAjax = /\s+/,
	rts = /([?&])_=[^&]*/,
	rurl = /^([\w\+\.\-]+:)(?:\/\/([^\/?#:]*)(?::(\d+))?)?/,

	// Keep a copy of the old load method
	_load = jQuery.fn.load,

	/* Prefilters
	 * 1) They are useful to introduce custom dataTypes (see ajax/jsonp.js for an example)
	 * 2) These are called:
	 *    - BEFORE asking for a transport
	 *    - AFTER param serialization (s.data is a string if s.processData is true)
	 * 3) key is the dataType
	 * 4) the catchall symbol "*" can be used
	 * 5) execution will start with transport dataType and THEN continue down to "*" if needed
	 */
	prefilters = {},

	/* Transports bindings
	 * 1) key is the dataType
	 * 2) the catchall symbol "*" can be used
	 * 3) selection will start with transport dataType and THEN go to "*" if needed
	 */
	transports = {},

	// Document location
	ajaxLocation,

	// Document location segments
	ajaxLocParts,
	
	// Avoid comment-prolog char sequence (#10098); must appease lint and evade compression
	allTypes = ["*/"] + ["*"];

// #8138, IE may throw an exception when accessing
// a field from window.location if document.domain has been set
try {
	ajaxLocation = location.href;
} catch( e ) {
	// Use the href attribute of an A element
	// since IE will modify it given document.location
	ajaxLocation = document.createElement( "a" );
	ajaxLocation.href = "";
	ajaxLocation = ajaxLocation.href;
}

// Segment location into parts
ajaxLocParts = rurl.exec( ajaxLocation.toLowerCase() ) || [];

// Base "constructor" for jQuery.ajaxPrefilter and jQuery.ajaxTransport
function addToPrefiltersOrTransports( structure ) {

	// dataTypeExpression is optional and defaults to "*"
	return function( dataTypeExpression, func ) {

		if ( typeof dataTypeExpression !== "string" ) {
			func = dataTypeExpression;
			dataTypeExpression = "*";
		}

		if ( jQuery.isFunction( func ) ) {
			var dataTypes = dataTypeExpression.toLowerCase().split( rspacesAjax ),
				i = 0,
				length = dataTypes.length,
				dataType,
				list,
				placeBefore;

			// For each dataType in the dataTypeExpression
			for(; i < length; i++ ) {
				dataType = dataTypes[ i ];
				// We control if we're asked to add before
				// any existing element
				placeBefore = /^\+/.test( dataType );
				if ( placeBefore ) {
					dataType = dataType.substr( 1 ) || "*";
				}
				list = structure[ dataType ] = structure[ dataType ] || [];
				// then we add to the structure accordingly
				list[ placeBefore ? "unshift" : "push" ]( func );
			}
		}
	};
}

// Base inspection function for prefilters and transports
function inspectPrefiltersOrTransports( structure, options, originalOptions, jqXHR,
		dataType /* internal */, inspected /* internal */ ) {

	dataType = dataType || options.dataTypes[ 0 ];
	inspected = inspected || {};

	inspected[ dataType ] = true;

	var list = structure[ dataType ],
		i = 0,
		length = list ? list.length : 0,
		executeOnly = ( structure === prefilters ),
		selection;

	for(; i < length && ( executeOnly || !selection ); i++ ) {
		selection = list[ i ]( options, originalOptions, jqXHR );
		// If we got redirected to another dataType
		// we try there if executing only and not done already
		if ( typeof selection === "string" ) {
			if ( !executeOnly || inspected[ selection ] ) {
				selection = undefined;
			} else {
				options.dataTypes.unshift( selection );
				selection = inspectPrefiltersOrTransports(
						structure, options, originalOptions, jqXHR, selection, inspected );
			}
		}
	}
	// If we're only executing or nothing was selected
	// we try the catchall dataType if not done already
	if ( ( executeOnly || !selection ) && !inspected[ "*" ] ) {
		selection = inspectPrefiltersOrTransports(
				structure, options, originalOptions, jqXHR, "*", inspected );
	}
	// unnecessary when only executing (prefilters)
	// but it'll be ignored by the caller in that case
	return selection;
}

// A special extend for ajax options
// that takes "flat" options (not to be deep extended)
// Fixes #9887
function ajaxExtend( target, src ) {
	var key, deep,
		flatOptions = jQuery.ajaxSettings.flatOptions || {};
	for( key in src ) {
		if ( src[ key ] !== undefined ) {
			( flatOptions[ key ] ? target : ( deep || ( deep = {} ) ) )[ key ] = src[ key ];
		}
	}
	if ( deep ) {
		jQuery.extend( true, target, deep );
	}
}

jQuery.fn.extend({
	load: function( url, params, callback ) {
		if ( typeof url !== "string" && _load ) {
			return _load.apply( this, arguments );

		// Don't do a request if no elements are being requested
		} else if ( !this.length ) {
			return this;
		}

		var off = url.indexOf( " " );
		if ( off >= 0 ) {
			var selector = url.slice( off, url.length );
			url = url.slice( 0, off );
		}

		// Default to a GET request
		var type = "GET";

		// If the second parameter was provided
		if ( params ) {
			// If it's a function
			if ( jQuery.isFunction( params ) ) {
				// We assume that it's the callback
				callback = params;
				params = undefined;

			// Otherwise, build a param string
			} else if ( typeof params === "object" ) {
				params = jQuery.param( params, jQuery.ajaxSettings.traditional );
				type = "POST";
			}
		}

		var self = this;

		// Request the remote document
		jQuery.ajax({
			url: url,
			type: type,
			dataType: "html",
			data: params,
			// Complete callback (responseText is used internally)
			complete: function( jqXHR, status, responseText ) {
				// Store the response as specified by the jqXHR object
				responseText = jqXHR.responseText;
				// If successful, inject the HTML into all the matched elements
				if ( jqXHR.isResolved() ) {
					// #4825: Get the actual response in case
					// a dataFilter is present in ajaxSettings
					jqXHR.done(function( r ) {
						responseText = r;
					});
					// See if a selector was specified
					self.html( selector ?
						// Create a dummy div to hold the results
						jQuery("<div>")
							// inject the contents of the document in, removing the scripts
							// to avoid any 'Permission Denied' errors in IE
							.append(responseText.replace(rscript, ""))

							// Locate the specified elements
							.find(selector) :

						// If not, just inject the full result
						responseText );
				}

				if ( callback ) {
					self.each( callback, [ responseText, status, jqXHR ] );
				}
			}
		});

		return this;
	},

	serialize: function() {
		return jQuery.param( this.serializeArray() );
	},

	serializeArray: function() {
		return this.map(function(){
			return this.elements ? jQuery.makeArray( this.elements ) : this;
		})
		.filter(function(){
			return this.name && !this.disabled &&
				( this.checked || rselectTextarea.test( this.nodeName ) ||
					rinput.test( this.type ) );
		})
		.map(function( i, elem ){
			var val = jQuery( this ).val();

			return val == null ?
				null :
				jQuery.isArray( val ) ?
					jQuery.map( val, function( val, i ){
						return { name: elem.name, value: val.replace( rCRLF, "\r\n" ) };
					}) :
					{ name: elem.name, value: val.replace( rCRLF, "\r\n" ) };
		}).get();
	}
});

// Attach a bunch of functions for handling common AJAX events
jQuery.each( "ajaxStart ajaxStop ajaxComplete ajaxError ajaxSuccess ajaxSend".split( " " ), function( i, o ){
	jQuery.fn[ o ] = function( f ){
		return this.bind( o, f );
	};
});

jQuery.each( [ "get", "post" ], function( i, method ) {
	jQuery[ method ] = function( url, data, callback, type ) {
		// shift arguments if data argument was omitted
		if ( jQuery.isFunction( data ) ) {
			type = type || callback;
			callback = data;
			data = undefined;
		}

		return jQuery.ajax({
			type: method,
			url: url,
			data: data,
			success: callback,
			dataType: type
		});
	};
});

jQuery.extend({

	getScript: function( url, callback ) {
		return jQuery.get( url, undefined, callback, "script" );
	},

	getJSON: function( url, data, callback ) {
		return jQuery.get( url, data, callback, "json" );
	},

	// Creates a full fledged settings object into target
	// with both ajaxSettings and settings fields.
	// If target is omitted, writes into ajaxSettings.
	ajaxSetup: function( target, settings ) {
		if ( settings ) {
			// Building a settings object
			ajaxExtend( target, jQuery.ajaxSettings );
		} else {
			// Extending ajaxSettings
			settings = target;
			target = jQuery.ajaxSettings;
		}
		ajaxExtend( target, settings );
		return target;
	},

	ajaxSettings: {
		url: ajaxLocation,
		isLocal: rlocalProtocol.test( ajaxLocParts[ 1 ] ),
		global: true,
		type: "GET",
		contentType: "application/x-www-form-urlencoded",
		processData: true,
		async: true,
		/*
		timeout: 0,
		data: null,
		dataType: null,
		username: null,
		password: null,
		cache: null,
		traditional: false,
		headers: {},
		*/

		accepts: {
			xml: "application/xml, text/xml",
			html: "text/html",
			text: "text/plain",
			json: "application/json, text/javascript",
			"*": allTypes
		},

		contents: {
			xml: /xml/,
			html: /html/,
			json: /json/
		},

		responseFields: {
			xml: "responseXML",
			text: "responseText"
		},

		// List of data converters
		// 1) key format is "source_type destination_type" (a single space in-between)
		// 2) the catchall symbol "*" can be used for source_type
		converters: {

			// Convert anything to text
			"* text": window.String,

			// Text to html (true = no transformation)
			"text html": true,

			// Evaluate text as a json expression
			"text json": jQuery.parseJSON,

			// Parse text as xml
			"text xml": jQuery.parseXML
		},

		// For options that shouldn't be deep extended:
		// you can add your own custom options here if
		// and when you create one that shouldn't be
		// deep extended (see ajaxExtend)
		flatOptions: {
			context: true,
			url: true
		}
	},

	ajaxPrefilter: addToPrefiltersOrTransports( prefilters ),
	ajaxTransport: addToPrefiltersOrTransports( transports ),

	// Main method
	ajax: function( url, options ) {

		// If url is an object, simulate pre-1.5 signature
		if ( typeof url === "object" ) {
			options = url;
			url = undefined;
		}

		// Force options to be an object
		options = options || {};

		var // Create the final options object
			s = jQuery.ajaxSetup( {}, options ),
			// Callbacks context
			callbackContext = s.context || s,
			// Context for global events
			// It's the callbackContext if one was provided in the options
			// and if it's a DOM node or a jQuery collection
			globalEventContext = callbackContext !== s &&
				( callbackContext.nodeType || callbackContext instanceof jQuery ) ?
						jQuery( callbackContext ) : jQuery.event,
			// Deferreds
			deferred = jQuery.Deferred(),
			completeDeferred = jQuery._Deferred(),
			// Status-dependent callbacks
			statusCode = s.statusCode || {},
			// ifModified key
			ifModifiedKey,
			// Headers (they are sent all at once)
			requestHeaders = {},
			requestHeadersNames = {},
			// Response headers
			responseHeadersString,
			responseHeaders,
			// transport
			transport,
			// timeout handle
			timeoutTimer,
			// Cross-domain detection vars
			parts,
			// The jqXHR state
			state = 0,
			// To know if global events are to be dispatched
			fireGlobals,
			// Loop variable
			i,
			// Fake xhr
			jqXHR = {

				readyState: 0,

				// Caches the header
				setRequestHeader: function( name, value ) {
					if ( !state ) {
						var lname = name.toLowerCase();
						name = requestHeadersNames[ lname ] = requestHeadersNames[ lname ] || name;
						requestHeaders[ name ] = value;
					}
					return this;
				},

				// Raw string
				getAllResponseHeaders: function() {
					return state === 2 ? responseHeadersString : null;
				},

				// Builds headers hashtable if needed
				getResponseHeader: function( key ) {
					var match;
					if ( state === 2 ) {
						if ( !responseHeaders ) {
							responseHeaders = {};
							while( ( match = rheaders.exec( responseHeadersString ) ) ) {
								responseHeaders[ match[1].toLowerCase() ] = match[ 2 ];
							}
						}
						match = responseHeaders[ key.toLowerCase() ];
					}
					return match === undefined ? null : match;
				},

				// Overrides response content-type header
				overrideMimeType: function( type ) {
					if ( !state ) {
						s.mimeType = type;
					}
					return this;
				},

				// Cancel the request
				abort: function( statusText ) {
					statusText = statusText || "abort";
					if ( transport ) {
						transport.abort( statusText );
					}
					done( 0, statusText );
					return this;
				}
			};

		// Callback for when everything is done
		// It is defined here because jslint complains if it is declared
		// at the end of the function (which would be more logical and readable)
		function done( status, nativeStatusText, responses, headers ) {

			// Called once
			if ( state === 2 ) {
				return;
			}

			// State is "done" now
			state = 2;

			// Clear timeout if it exists
			if ( timeoutTimer ) {
				clearTimeout( timeoutTimer );
			}

			// Dereference transport for early garbage collection
			// (no matter how long the jqXHR object will be used)
			transport = undefined;

			// Cache response headers
			responseHeadersString = headers || "";

			// Set readyState
			jqXHR.readyState = status > 0 ? 4 : 0;

			var isSuccess,
				success,
				error,
				statusText = nativeStatusText,
				response = responses ? ajaxHandleResponses( s, jqXHR, responses ) : undefined,
				lastModified,
				etag;

			// If successful, handle type chaining
			if ( status >= 200 && status < 300 || status === 304 ) {

				// Set the If-Modified-Since and/or If-None-Match header, if in ifModified mode.
				if ( s.ifModified ) {

					if ( ( lastModified = jqXHR.getResponseHeader( "Last-Modified" ) ) ) {
						jQuery.lastModified[ ifModifiedKey ] = lastModified;
					}
					if ( ( etag = jqXHR.getResponseHeader( "Etag" ) ) ) {
						jQuery.etag[ ifModifiedKey ] = etag;
					}
				}

				// If not modified
				if ( status === 304 ) {

					statusText = "notmodified";
					isSuccess = true;

				// If we have data
				} else {

					try {
						success = ajaxConvert( s, response );
						statusText = "success";
						isSuccess = true;
					} catch(e) {
						// We have a parsererror
						statusText = "parsererror";
						error = e;
					}
				}
			} else {
				// We extract error from statusText
				// then normalize statusText and status for non-aborts
				error = statusText;
				if( !statusText || status ) {
					statusText = "error";
					if ( status < 0 ) {
						status = 0;
					}
				}
			}

			// Set data for the fake xhr object
			jqXHR.status = status;
			jqXHR.statusText = "" + ( nativeStatusText || statusText );

			// Success/Error
			if ( isSuccess ) {
				deferred.resolveWith( callbackContext, [ success, statusText, jqXHR ] );
			} else {
				deferred.rejectWith( callbackContext, [ jqXHR, statusText, error ] );
			}

			// Status-dependent callbacks
			jqXHR.statusCode( statusCode );
			statusCode = undefined;

			if ( fireGlobals ) {
				globalEventContext.trigger( "ajax" + ( isSuccess ? "Success" : "Error" ),
						[ jqXHR, s, isSuccess ? success : error ] );
			}

			// Complete
			completeDeferred.resolveWith( callbackContext, [ jqXHR, statusText ] );

			if ( fireGlobals ) {
				globalEventContext.trigger( "ajaxComplete", [ jqXHR, s ] );
				// Handle the global AJAX counter
				if ( !( --jQuery.active ) ) {
					jQuery.event.trigger( "ajaxStop" );
				}
			}
		}

		// Attach deferreds
		deferred.promise( jqXHR );
		jqXHR.success = jqXHR.done;
		jqXHR.error = jqXHR.fail;
		jqXHR.complete = completeDeferred.done;

		// Status-dependent callbacks
		jqXHR.statusCode = function( map ) {
			if ( map ) {
				var tmp;
				if ( state < 2 ) {
					for( tmp in map ) {
						statusCode[ tmp ] = [ statusCode[tmp], map[tmp] ];
					}
				} else {
					tmp = map[ jqXHR.status ];
					jqXHR.then( tmp, tmp );
				}
			}
			return this;
		};

		// Remove hash character (#7531: and string promotion)
		// Add protocol if not provided (#5866: IE7 issue with protocol-less urls)
		// We also use the url parameter if available
		s.url = ( ( url || s.url ) + "" ).replace( rhash, "" ).replace( rprotocol, ajaxLocParts[ 1 ] + "//" );

		// Extract dataTypes list
		s.dataTypes = jQuery.trim( s.dataType || "*" ).toLowerCase().split( rspacesAjax );

		// Determine if a cross-domain request is in order
		if ( s.crossDomain == null ) {
			parts = rurl.exec( s.url.toLowerCase() );
			s.crossDomain = !!( parts &&
				( parts[ 1 ] != ajaxLocParts[ 1 ] || parts[ 2 ] != ajaxLocParts[ 2 ] ||
					( parts[ 3 ] || ( parts[ 1 ] === "http:" ? 80 : 443 ) ) !=
						( ajaxLocParts[ 3 ] || ( ajaxLocParts[ 1 ] === "http:" ? 80 : 443 ) ) )
			);
		}

		// Convert data if not already a string
		if ( s.data && s.processData && typeof s.data !== "string" ) {
			s.data = jQuery.param( s.data, s.traditional );
		}

		// Apply prefilters
		inspectPrefiltersOrTransports( prefilters, s, options, jqXHR );

		// If request was aborted inside a prefiler, stop there
		if ( state === 2 ) {
			return false;
		}

		// We can fire global events as of now if asked to
		fireGlobals = s.global;

		// Uppercase the type
		s.type = s.type.toUpperCase();

		// Determine if request has content
		s.hasContent = !rnoContent.test( s.type );

		// Watch for a new set of requests
		if ( fireGlobals && jQuery.active++ === 0 ) {
			jQuery.event.trigger( "ajaxStart" );
		}

		// More options handling for requests with no content
		if ( !s.hasContent ) {

			// If data is available, append data to url
			if ( s.data ) {
				s.url += ( rquery.test( s.url ) ? "&" : "?" ) + s.data;
				// #9682: remove data so that it's not used in an eventual retry
				delete s.data;
			}

			// Get ifModifiedKey before adding the anti-cache parameter
			ifModifiedKey = s.url;

			// Add anti-cache in url if needed
			if ( s.cache === false ) {

				var ts = jQuery.now(),
					// try replacing _= if it is there
					ret = s.url.replace( rts, "$1_=" + ts );

				// if nothing was replaced, add timestamp to the end
				s.url = ret + ( (ret === s.url ) ? ( rquery.test( s.url ) ? "&" : "?" ) + "_=" + ts : "" );
			}
		}

		// Set the correct header, if data is being sent
		if ( s.data && s.hasContent && s.contentType !== false || options.contentType ) {
			jqXHR.setRequestHeader( "Content-Type", s.contentType );
		}

		// Set the If-Modified-Since and/or If-None-Match header, if in ifModified mode.
		if ( s.ifModified ) {
			ifModifiedKey = ifModifiedKey || s.url;
			if ( jQuery.lastModified[ ifModifiedKey ] ) {
				jqXHR.setRequestHeader( "If-Modified-Since", jQuery.lastModified[ ifModifiedKey ] );
			}
			if ( jQuery.etag[ ifModifiedKey ] ) {
				jqXHR.setRequestHeader( "If-None-Match", jQuery.etag[ ifModifiedKey ] );
			}
		}

		// Set the Accepts header for the server, depending on the dataType
		jqXHR.setRequestHeader(
			"Accept",
			s.dataTypes[ 0 ] && s.accepts[ s.dataTypes[0] ] ?
				s.accepts[ s.dataTypes[0] ] + ( s.dataTypes[ 0 ] !== "*" ? ", " + allTypes + "; q=0.01" : "" ) :
				s.accepts[ "*" ]
		);

		// Check for headers option
		for ( i in s.headers ) {
			jqXHR.setRequestHeader( i, s.headers[ i ] );
		}

		// Allow custom headers/mimetypes and early abort
		if ( s.beforeSend && ( s.beforeSend.call( callbackContext, jqXHR, s ) === false || state === 2 ) ) {
				// Abort if not done already
				jqXHR.abort();
				return false;

		}

		// Install callbacks on deferreds
		for ( i in { success: 1, error: 1, complete: 1 } ) {
			jqXHR[ i ]( s[ i ] );
		}

		// Get transport
		transport = inspectPrefiltersOrTransports( transports, s, options, jqXHR );

		// If no transport, we auto-abort
		if ( !transport ) {
			done( -1, "No Transport" );
		} else {
			jqXHR.readyState = 1;
			// Send global event
			if ( fireGlobals ) {
				globalEventContext.trigger( "ajaxSend", [ jqXHR, s ] );
			}
			// Timeout
			if ( s.async && s.timeout > 0 ) {
				timeoutTimer = setTimeout( function(){
					jqXHR.abort( "timeout" );
				}, s.timeout );
			}

			try {
				state = 1;
				transport.send( requestHeaders, done );
			} catch (e) {
				// Propagate exception as error if not done
				if ( state < 2 ) {
					done( -1, e );
				// Simply rethrow otherwise
				} else {
					jQuery.error( e );
				}
			}
		}

		return jqXHR;
	},

	// Serialize an array of form elements or a set of
	// key/values into a query string
	param: function( a, traditional ) {
		var s = [],
			add = function( key, value ) {
				// If value is a function, invoke it and return its value
				value = jQuery.isFunction( value ) ? value() : value;
				s[ s.length ] = encodeURIComponent( key ) + "=" + encodeURIComponent( value );
			};

		// Set traditional to true for jQuery <= 1.3.2 behavior.
		if ( traditional === undefined ) {
			traditional = jQuery.ajaxSettings.traditional;
		}

		// If an array was passed in, assume that it is an array of form elements.
		if ( jQuery.isArray( a ) || ( a.jquery && !jQuery.isPlainObject( a ) ) ) {
			// Serialize the form elements
			jQuery.each( a, function() {
				add( this.name, this.value );
			});

		} else {
			// If traditional, encode the "old" way (the way 1.3.2 or older
			// did it), otherwise encode params recursively.
			for ( var prefix in a ) {
				buildParams( prefix, a[ prefix ], traditional, add );
			}
		}

		// Return the resulting serialization
		return s.join( "&" ).replace( r20, "+" );
	}
});

function buildParams( prefix, obj, traditional, add ) {
	if ( jQuery.isArray( obj ) ) {
		// Serialize array item.
		jQuery.each( obj, function( i, v ) {
			if ( traditional || rbracket.test( prefix ) ) {
				// Treat each array item as a scalar.
				add( prefix, v );

			} else {
				// If array item is non-scalar (array or object), encode its
				// numeric index to resolve deserialization ambiguity issues.
				// Note that rack (as of 1.0.0) can't currently deserialize
				// nested arrays properly, and attempting to do so may cause
				// a server error. Possible fixes are to modify rack's
				// deserialization algorithm or to provide an option or flag
				// to force array serialization to be shallow.
				buildParams( prefix + "[" + ( typeof v === "object" || jQuery.isArray(v) ? i : "" ) + "]", v, traditional, add );
			}
		});

	} else if ( !traditional && obj != null && typeof obj === "object" ) {
		// Serialize object item.
		for ( var name in obj ) {
			buildParams( prefix + "[" + name + "]", obj[ name ], traditional, add );
		}

	} else {
		// Serialize scalar item.
		add( prefix, obj );
	}
}

// This is still on the jQuery object... for now
// Want to move this to jQuery.ajax some day
jQuery.extend({

	// Counter for holding the number of active queries
	active: 0,

	// Last-Modified header cache for next request
	lastModified: {},
	etag: {}

});

/* Handles responses to an ajax request:
 * - sets all responseXXX fields accordingly
 * - finds the right dataType (mediates between content-type and expected dataType)
 * - returns the corresponding response
 */
function ajaxHandleResponses( s, jqXHR, responses ) {

	var contents = s.contents,
		dataTypes = s.dataTypes,
		responseFields = s.responseFields,
		ct,
		type,
		finalDataType,
		firstDataType;

	// Fill responseXXX fields
	for( type in responseFields ) {
		if ( type in responses ) {
			jqXHR[ responseFields[type] ] = responses[ type ];
		}
	}

	// Remove auto dataType and get content-type in the process
	while( dataTypes[ 0 ] === "*" ) {
		dataTypes.shift();
		if ( ct === undefined ) {
			ct = s.mimeType || jqXHR.getResponseHeader( "content-type" );
		}
	}

	// Check if we're dealing with a known content-type
	if ( ct ) {
		for ( type in contents ) {
			if ( contents[ type ] && contents[ type ].test( ct ) ) {
				dataTypes.unshift( type );
				break;
			}
		}
	}

	// Check to see if we have a response for the expected dataType
	if ( dataTypes[ 0 ] in responses ) {
		finalDataType = dataTypes[ 0 ];
	} else {
		// Try convertible dataTypes
		for ( type in responses ) {
			if ( !dataTypes[ 0 ] || s.converters[ type + " " + dataTypes[0] ] ) {
				finalDataType = type;
				break;
			}
			if ( !firstDataType ) {
				firstDataType = type;
			}
		}
		// Or just use first one
		finalDataType = finalDataType || firstDataType;
	}

	// If we found a dataType
	// We add the dataType to the list if needed
	// and return the corresponding response
	if ( finalDataType ) {
		if ( finalDataType !== dataTypes[ 0 ] ) {
			dataTypes.unshift( finalDataType );
		}
		return responses[ finalDataType ];
	}
}

// Chain conversions given the request and the original response
function ajaxConvert( s, response ) {

	// Apply the dataFilter if provided
	if ( s.dataFilter ) {
		response = s.dataFilter( response, s.dataType );
	}

	var dataTypes = s.dataTypes,
		converters = {},
		i,
		key,
		length = dataTypes.length,
		tmp,
		// Current and previous dataTypes
		current = dataTypes[ 0 ],
		prev,
		// Conversion expression
		conversion,
		// Conversion function
		conv,
		// Conversion functions (transitive conversion)
		conv1,
		conv2;

	// For each dataType in the chain
	for( i = 1; i < length; i++ ) {

		// Create converters map
		// with lowercased keys
		if ( i === 1 ) {
			for( key in s.converters ) {
				if( typeof key === "string" ) {
					converters[ key.toLowerCase() ] = s.converters[ key ];
				}
			}
		}

		// Get the dataTypes
		prev = current;
		current = dataTypes[ i ];

		// If current is auto dataType, update it to prev
		if( current === "*" ) {
			current = prev;
		// If no auto and dataTypes are actually different
		} else if ( prev !== "*" && prev !== current ) {

			// Get the converter
			conversion = prev + " " + current;
			conv = converters[ conversion ] || converters[ "* " + current ];

			// If there is no direct converter, search transitively
			if ( !conv ) {
				conv2 = undefined;
				for( conv1 in converters ) {
					tmp = conv1.split( " " );
					if ( tmp[ 0 ] === prev || tmp[ 0 ] === "*" ) {
						conv2 = converters[ tmp[1] + " " + current ];
						if ( conv2 ) {
							conv1 = converters[ conv1 ];
							if ( conv1 === true ) {
								conv = conv2;
							} else if ( conv2 === true ) {
								conv = conv1;
							}
							break;
						}
					}
				}
			}
			// If we found no converter, dispatch an error
			if ( !( conv || conv2 ) ) {
				jQuery.error( "No conversion from " + conversion.replace(" "," to ") );
			}
			// If found converter is not an equivalence
			if ( conv !== true ) {
				// Convert with 1 or 2 converters accordingly
				response = conv ? conv( response ) : conv2( conv1(response) );
			}
		}
	}
	return response;
}




var jsc = jQuery.now(),
	jsre = /(\=)\?(&|$)|\?\?/i;

// Default jsonp settings
jQuery.ajaxSetup({
	jsonp: "callback",
	jsonpCallback: function() {
		return jQuery.expando + "_" + ( jsc++ );
	}
});

// Detect, normalize options and install callbacks for jsonp requests
jQuery.ajaxPrefilter( "json jsonp", function( s, originalSettings, jqXHR ) {

	var inspectData = s.contentType === "application/x-www-form-urlencoded" &&
		( typeof s.data === "string" );

	if ( s.dataTypes[ 0 ] === "jsonp" ||
		s.jsonp !== false && ( jsre.test( s.url ) ||
				inspectData && jsre.test( s.data ) ) ) {

		var responseContainer,
			jsonpCallback = s.jsonpCallback =
				jQuery.isFunction( s.jsonpCallback ) ? s.jsonpCallback() : s.jsonpCallback,
			previous = window[ jsonpCallback ],
			url = s.url,
			data = s.data,
			replace = "$1" + jsonpCallback + "$2";

		if ( s.jsonp !== false ) {
			url = url.replace( jsre, replace );
			if ( s.url === url ) {
				if ( inspectData ) {
					data = data.replace( jsre, replace );
				}
				if ( s.data === data ) {
					// Add callback manually
					url += (/\?/.test( url ) ? "&" : "?") + s.jsonp + "=" + jsonpCallback;
				}
			}
		}

		s.url = url;
		s.data = data;

		// Install callback
		window[ jsonpCallback ] = function( response ) {
			responseContainer = [ response ];
		};

		// Clean-up function
		jqXHR.always(function() {
			// Set callback back to previous value
			window[ jsonpCallback ] = previous;
			// Call if it was a function and we have a response
			if ( responseContainer && jQuery.isFunction( previous ) ) {
				window[ jsonpCallback ]( responseContainer[ 0 ] );
			}
		});

		// Use data converter to retrieve json after script execution
		s.converters["script json"] = function() {
			if ( !responseContainer ) {
				jQuery.error( jsonpCallback + " was not called" );
			}
			return responseContainer[ 0 ];
		};

		// force json dataType
		s.dataTypes[ 0 ] = "json";

		// Delegate to script
		return "script";
	}
});




// Install script dataType
jQuery.ajaxSetup({
	accepts: {
		script: "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript"
	},
	contents: {
		script: /javascript|ecmascript/
	},
	converters: {
		"text script": function( text ) {
			jQuery.globalEval( text );
			return text;
		}
	}
});

// Handle cache's special case and global
jQuery.ajaxPrefilter( "script", function( s ) {
	if ( s.cache === undefined ) {
		s.cache = false;
	}
	if ( s.crossDomain ) {
		s.type = "GET";
		s.global = false;
	}
});

// Bind script tag hack transport
jQuery.ajaxTransport( "script", function(s) {

	// This transport only deals with cross domain requests
	if ( s.crossDomain ) {

		var script,
			head = document.head || document.getElementsByTagName( "head" )[0] || document.documentElement;

		return {

			send: function( _, callback ) {

				script = document.createElement( "script" );

				script.async = "async";

				if ( s.scriptCharset ) {
					script.charset = s.scriptCharset;
				}

				script.src = s.url;

				// Attach handlers for all browsers
				script.onload = script.onreadystatechange = function( _, isAbort ) {

					if ( isAbort || !script.readyState || /loaded|complete/.test( script.readyState ) ) {

						// Handle memory leak in IE
						script.onload = script.onreadystatechange = null;

						// Remove the script
						if ( head && script.parentNode ) {
							head.removeChild( script );
						}

						// Dereference the script
						script = undefined;

						// Callback if not abort
						if ( !isAbort ) {
							callback( 200, "success" );
						}
					}
				};
				// Use insertBefore instead of appendChild  to circumvent an IE6 bug.
				// This arises when a base node is used (#2709 and #4378).
				head.insertBefore( script, head.firstChild );
			},

			abort: function() {
				if ( script ) {
					script.onload( 0, 1 );
				}
			}
		};
	}
});




var // #5280: Internet Explorer will keep connections alive if we don't abort on unload
	xhrOnUnloadAbort = window.ActiveXObject ? function() {
		// Abort all pending requests
		for ( var key in xhrCallbacks ) {
			xhrCallbacks[ key ]( 0, 1 );
		}
	} : false,
	xhrId = 0,
	xhrCallbacks;

// Functions to create xhrs
function createStandardXHR() {
	try {
		return new window.XMLHttpRequest();
	} catch( e ) {}
}

function createActiveXHR() {
	try {
		return new window.ActiveXObject( "Microsoft.XMLHTTP" );
	} catch( e ) {}
}

// Create the request object
// (This is still attached to ajaxSettings for backward compatibility)
jQuery.ajaxSettings.xhr = window.ActiveXObject ?
	/* Microsoft failed to properly
	 * implement the XMLHttpRequest in IE7 (can't request local files),
	 * so we use the ActiveXObject when it is available
	 * Additionally XMLHttpRequest can be disabled in IE7/IE8 so
	 * we need a fallback.
	 */
	function() {
		return !this.isLocal && createStandardXHR() || createActiveXHR();
	} :
	// For all other browsers, use the standard XMLHttpRequest object
	createStandardXHR;

// Determine support properties
(function( xhr ) {
	jQuery.extend( jQuery.support, {
		ajax: !!xhr,
		cors: !!xhr && ( "withCredentials" in xhr )
	});
})( jQuery.ajaxSettings.xhr() );

// Create transport if the browser can provide an xhr
if ( jQuery.support.ajax ) {

	jQuery.ajaxTransport(function( s ) {
		// Cross domain only allowed if supported through XMLHttpRequest
		if ( !s.crossDomain || jQuery.support.cors ) {

			var callback;

			return {
				send: function( headers, complete ) {

					// Get a new xhr
					var xhr = s.xhr(),
						handle,
						i;

					// Open the socket
					// Passing null username, generates a login popup on Opera (#2865)
					if ( s.username ) {
						xhr.open( s.type, s.url, s.async, s.username, s.password );
					} else {
						xhr.open( s.type, s.url, s.async );
					}

					// Apply custom fields if provided
					if ( s.xhrFields ) {
						for ( i in s.xhrFields ) {
							xhr[ i ] = s.xhrFields[ i ];
						}
					}

					// Override mime type if needed
					if ( s.mimeType && xhr.overrideMimeType ) {
						xhr.overrideMimeType( s.mimeType );
					}

					// X-Requested-With header
					// For cross-domain requests, seeing as conditions for a preflight are
					// akin to a jigsaw puzzle, we simply never set it to be sure.
					// (it can always be set on a per-request basis or even using ajaxSetup)
					// For same-domain requests, won't change header if already provided.
					if ( !s.crossDomain && !headers["X-Requested-With"] ) {
						headers[ "X-Requested-With" ] = "XMLHttpRequest";
					}

					// Need an extra try/catch for cross domain requests in Firefox 3
					try {
						for ( i in headers ) {
							xhr.setRequestHeader( i, headers[ i ] );
						}
					} catch( _ ) {}

					// Do send the request
					// This may raise an exception which is actually
					// handled in jQuery.ajax (so no try/catch here)
					xhr.send( ( s.hasContent && s.data ) || null );

					// Listener
					callback = function( _, isAbort ) {

						var status,
							statusText,
							responseHeaders,
							responses,
							xml;

						// Firefox throws exceptions when accessing properties
						// of an xhr when a network error occured
						// http://helpful.knobs-dials.com/index.php/Component_returned_failure_code:_0x80040111_(NS_ERROR_NOT_AVAILABLE)
						try {

							// Was never called and is aborted or complete
							if ( callback && ( isAbort || xhr.readyState === 4 ) ) {

								// Only called once
								callback = undefined;

								// Do not keep as active anymore
								if ( handle ) {
									xhr.onreadystatechange = jQuery.noop;
									if ( xhrOnUnloadAbort ) {
										delete xhrCallbacks[ handle ];
									}
								}

								// If it's an abort
								if ( isAbort ) {
									// Abort it manually if needed
									if ( xhr.readyState !== 4 ) {
										xhr.abort();
									}
								} else {
									status = xhr.status;
									responseHeaders = xhr.getAllResponseHeaders();
									responses = {};
									xml = xhr.responseXML;

									// Construct response list
									if ( xml && xml.documentElement /* #4958 */ ) {
										responses.xml = xml;
									}
									responses.text = xhr.responseText;

									// Firefox throws an exception when accessing
									// statusText for faulty cross-domain requests
									try {
										statusText = xhr.statusText;
									} catch( e ) {
										// We normalize with Webkit giving an empty statusText
										statusText = "";
									}

									// Filter status for non standard behaviors

									// If the request is local and we have data: assume a success
									// (success with no data won't get notified, that's the best we
									// can do given current implementations)
									if ( !status && s.isLocal && !s.crossDomain ) {
										status = responses.text ? 200 : 404;
									// IE - #1450: sometimes returns 1223 when it should be 204
									} else if ( status === 1223 ) {
										status = 204;
									}
								}
							}
						} catch( firefoxAccessException ) {
							if ( !isAbort ) {
								complete( -1, firefoxAccessException );
							}
						}

						// Call complete if needed
						if ( responses ) {
							complete( status, statusText, responses, responseHeaders );
						}
					};

					// if we're in sync mode or it's in cache
					// and has been retrieved directly (IE6 & IE7)
					// we need to manually fire the callback
					if ( !s.async || xhr.readyState === 4 ) {
						callback();
					} else {
						handle = ++xhrId;
						if ( xhrOnUnloadAbort ) {
							// Create the active xhrs callbacks list if needed
							// and attach the unload handler
							if ( !xhrCallbacks ) {
								xhrCallbacks = {};
								jQuery( window ).unload( xhrOnUnloadAbort );
							}
							// Add to list of active xhrs callbacks
							xhrCallbacks[ handle ] = callback;
						}
						xhr.onreadystatechange = callback;
					}
				},

				abort: function() {
					if ( callback ) {
						callback(0,1);
					}
				}
			};
		}
	});
}




var elemdisplay = {},
	iframe, iframeDoc,
	rfxtypes = /^(?:toggle|show|hide)$/,
	rfxnum = /^([+\-]=)?([\d+.\-]+)([a-z%]*)$/i,
	timerId,
	fxAttrs = [
		// height animations
		[ "height", "marginTop", "marginBottom", "paddingTop", "paddingBottom" ],
		// width animations
		[ "width", "marginLeft", "marginRight", "paddingLeft", "paddingRight" ],
		// opacity animations
		[ "opacity" ]
	],
	fxNow;

jQuery.fn.extend({
	show: function( speed, easing, callback ) {
		var elem, display;

		if ( speed || speed === 0 ) {
			return this.animate( genFx("show", 3), speed, easing, callback);

		} else {
			for ( var i = 0, j = this.length; i < j; i++ ) {
				elem = this[i];

				if ( elem.style ) {
					display = elem.style.display;

					// Reset the inline display of this element to learn if it is
					// being hidden by cascaded rules or not
					if ( !jQuery._data(elem, "olddisplay") && display === "none" ) {
						display = elem.style.display = "";
					}

					// Set elements which have been overridden with display: none
					// in a stylesheet to whatever the default browser style is
					// for such an element
					if ( display === "" && jQuery.css( elem, "display" ) === "none" ) {
						jQuery._data(elem, "olddisplay", defaultDisplay(elem.nodeName));
					}
				}
			}

			// Set the display of most of the elements in a second loop
			// to avoid the constant reflow
			for ( i = 0; i < j; i++ ) {
				elem = this[i];

				if ( elem.style ) {
					display = elem.style.display;

					if ( display === "" || display === "none" ) {
						elem.style.display = jQuery._data(elem, "olddisplay") || "";
					}
				}
			}

			return this;
		}
	},

	hide: function( speed, easing, callback ) {
		if ( speed || speed === 0 ) {
			return this.animate( genFx("hide", 3), speed, easing, callback);

		} else {
			for ( var i = 0, j = this.length; i < j; i++ ) {
				if ( this[i].style ) {
					var display = jQuery.css( this[i], "display" );

					if ( display !== "none" && !jQuery._data( this[i], "olddisplay" ) ) {
						jQuery._data( this[i], "olddisplay", display );
					}
				}
			}

			// Set the display of the elements in a second loop
			// to avoid the constant reflow
			for ( i = 0; i < j; i++ ) {
				if ( this[i].style ) {
					this[i].style.display = "none";
				}
			}

			return this;
		}
	},

	// Save the old toggle function
	_toggle: jQuery.fn.toggle,

	toggle: function( fn, fn2, callback ) {
		var bool = typeof fn === "boolean";

		if ( jQuery.isFunction(fn) && jQuery.isFunction(fn2) ) {
			this._toggle.apply( this, arguments );

		} else if ( fn == null || bool ) {
			this.each(function() {
				var state = bool ? fn : jQuery(this).is(":hidden");
				jQuery(this)[ state ? "show" : "hide" ]();
			});

		} else {
			this.animate(genFx("toggle", 3), fn, fn2, callback);
		}

		return this;
	},

	fadeTo: function( speed, to, easing, callback ) {
		return this.filter(":hidden").css("opacity", 0).show().end()
					.animate({opacity: to}, speed, easing, callback);
	},

	animate: function( prop, speed, easing, callback ) {
		var optall = jQuery.speed(speed, easing, callback);

		if ( jQuery.isEmptyObject( prop ) ) {
			return this.each( optall.complete, [ false ] );
		}

		// Do not change referenced properties as per-property easing will be lost
		prop = jQuery.extend( {}, prop );

		return this[ optall.queue === false ? "each" : "queue" ](function() {
			// XXX 'this' does not always have a nodeName when running the
			// test suite

			if ( optall.queue === false ) {
				jQuery._mark( this );
			}

			var opt = jQuery.extend( {}, optall ),
				isElement = this.nodeType === 1,
				hidden = isElement && jQuery(this).is(":hidden"),
				name, val, p,
				display, e,
				parts, start, end, unit;

			// will store per property easing and be used to determine when an animation is complete
			opt.animatedProperties = {};

			for ( p in prop ) {

				// property name normalization
				name = jQuery.camelCase( p );
				if ( p !== name ) {
					prop[ name ] = prop[ p ];
					delete prop[ p ];
				}

				val = prop[ name ];

				// easing resolution: per property > opt.specialEasing > opt.easing > 'swing' (default)
				if ( jQuery.isArray( val ) ) {
					opt.animatedProperties[ name ] = val[ 1 ];
					val = prop[ name ] = val[ 0 ];
				} else {
					opt.animatedProperties[ name ] = opt.specialEasing && opt.specialEasing[ name ] || opt.easing || 'swing';
				}

				if ( val === "hide" && hidden || val === "show" && !hidden ) {
					return opt.complete.call( this );
				}

				if ( isElement && ( name === "height" || name === "width" ) ) {
					// Make sure that nothing sneaks out
					// Record all 3 overflow attributes because IE does not
					// change the overflow attribute when overflowX and
					// overflowY are set to the same value
					opt.overflow = [ this.style.overflow, this.style.overflowX, this.style.overflowY ];

					// Set display property to inline-block for height/width
					// animations on inline elements that are having width/height
					// animated
					if ( jQuery.css( this, "display" ) === "inline" &&
							jQuery.css( this, "float" ) === "none" ) {
						if ( !jQuery.support.inlineBlockNeedsLayout ) {
							this.style.display = "inline-block";

						} else {
							display = defaultDisplay( this.nodeName );

							// inline-level elements accept inline-block;
							// block-level elements need to be inline with layout
							if ( display === "inline" ) {
								this.style.display = "inline-block";

							} else {
								this.style.display = "inline";
								this.style.zoom = 1;
							}
						}
					}
				}
			}

			if ( opt.overflow != null ) {
				this.style.overflow = "hidden";
			}

			for ( p in prop ) {
				e = new jQuery.fx( this, opt, p );
				val = prop[ p ];

				if ( rfxtypes.test(val) ) {
					e[ val === "toggle" ? hidden ? "show" : "hide" : val ]();

				} else {
					parts = rfxnum.exec( val );
					start = e.cur();

					if ( parts ) {
						end = parseFloat( parts[2] );
						unit = parts[3] || ( jQuery.cssNumber[ p ] ? "" : "px" );

						// We need to compute starting value
						if ( unit !== "px" ) {
							jQuery.style( this, p, (end || 1) + unit);
							start = ((end || 1) / e.cur()) * start;
							jQuery.style( this, p, start + unit);
						}

						// If a +=/-= token was provided, we're doing a relative animation
						if ( parts[1] ) {
							end = ( (parts[ 1 ] === "-=" ? -1 : 1) * end ) + start;
						}

						e.custom( start, end, unit );

					} else {
						e.custom( start, val, "" );
					}
				}
			}

			// For JS strict compliance
			return true;
		});
	},

	stop: function( clearQueue, gotoEnd ) {
		if ( clearQueue ) {
			this.queue([]);
		}

		this.each(function() {
			var timers = jQuery.timers,
				i = timers.length;
			// clear marker counters if we know they won't be
			if ( !gotoEnd ) {
				jQuery._unmark( true, this );
			}
			while ( i-- ) {
				if ( timers[i].elem === this ) {
					if (gotoEnd) {
						// force the next step to be the last
						timers[i](true);
					}

					timers.splice(i, 1);
				}
			}
		});

		// start the next in the queue if the last step wasn't forced
		if ( !gotoEnd ) {
			this.dequeue();
		}

		return this;
	}

});

// Animations created synchronously will run synchronously
function createFxNow() {
	setTimeout( clearFxNow, 0 );
	return ( fxNow = jQuery.now() );
}

function clearFxNow() {
	fxNow = undefined;
}

// Generate parameters to create a standard animation
function genFx( type, num ) {
	var obj = {};

	jQuery.each( fxAttrs.concat.apply([], fxAttrs.slice(0,num)), function() {
		obj[ this ] = type;
	});

	return obj;
}

// Generate shortcuts for custom animations
jQuery.each({
	slideDown: genFx("show", 1),
	slideUp: genFx("hide", 1),
	slideToggle: genFx("toggle", 1),
	fadeIn: { opacity: "show" },
	fadeOut: { opacity: "hide" },
	fadeToggle: { opacity: "toggle" }
}, function( name, props ) {
	jQuery.fn[ name ] = function( speed, easing, callback ) {
		return this.animate( props, speed, easing, callback );
	};
});

jQuery.extend({
	speed: function( speed, easing, fn ) {
		var opt = speed && typeof speed === "object" ? jQuery.extend({}, speed) : {
			complete: fn || !fn && easing ||
				jQuery.isFunction( speed ) && speed,
			duration: speed,
			easing: fn && easing || easing && !jQuery.isFunction(easing) && easing
		};

		opt.duration = jQuery.fx.off ? 0 : typeof opt.duration === "number" ? opt.duration :
			opt.duration in jQuery.fx.speeds ? jQuery.fx.speeds[opt.duration] : jQuery.fx.speeds._default;

		// Queueing
		opt.old = opt.complete;
		opt.complete = function( noUnmark ) {
			if ( jQuery.isFunction( opt.old ) ) {
				opt.old.call( this );
			}

			if ( opt.queue !== false ) {
				jQuery.dequeue( this );
			} else if ( noUnmark !== false ) {
				jQuery._unmark( this );
			}
		};

		return opt;
	},

	easing: {
		linear: function( p, n, firstNum, diff ) {
			return firstNum + diff * p;
		},
		swing: function( p, n, firstNum, diff ) {
			return ((-Math.cos(p*Math.PI)/2) + 0.5) * diff + firstNum;
		}
	},

	timers: [],

	fx: function( elem, options, prop ) {
		this.options = options;
		this.elem = elem;
		this.prop = prop;

		options.orig = options.orig || {};
	}

});

jQuery.fx.prototype = {
	// Simple function for setting a style value
	update: function() {
		if ( this.options.step ) {
			this.options.step.call( this.elem, this.now, this );
		}

		(jQuery.fx.step[this.prop] || jQuery.fx.step._default)( this );
	},

	// Get the current size
	cur: function() {
		if ( this.elem[this.prop] != null && (!this.elem.style || this.elem.style[this.prop] == null) ) {
			return this.elem[ this.prop ];
		}

		var parsed,
			r = jQuery.css( this.elem, this.prop );
		// Empty strings, null, undefined and "auto" are converted to 0,
		// complex values such as "rotate(1rad)" are returned as is,
		// simple values such as "10px" are parsed to Float.
		return isNaN( parsed = parseFloat( r ) ) ? !r || r === "auto" ? 0 : r : parsed;
	},

	// Start an animation from one number to another
	custom: function( from, to, unit ) {
		var self = this,
			fx = jQuery.fx;

		this.startTime = fxNow || createFxNow();
		this.start = from;
		this.end = to;
		this.unit = unit || this.unit || ( jQuery.cssNumber[ this.prop ] ? "" : "px" );
		this.now = this.start;
		this.pos = this.state = 0;

		function t( gotoEnd ) {
			return self.step(gotoEnd);
		}

		t.elem = this.elem;

		if ( t() && jQuery.timers.push(t) && !timerId ) {
			timerId = setInterval( fx.tick, fx.interval );
		}
	},

	// Simple 'show' function
	show: function() {
		// Remember where we started, so that we can go back to it later
		this.options.orig[this.prop] = jQuery.style( this.elem, this.prop );
		this.options.show = true;

		// Begin the animation
		// Make sure that we start at a small width/height to avoid any
		// flash of content
		this.custom(this.prop === "width" || this.prop === "height" ? 1 : 0, this.cur());

		// Start by showing the element
		jQuery( this.elem ).show();
	},

	// Simple 'hide' function
	hide: function() {
		// Remember where we started, so that we can go back to it later
		this.options.orig[this.prop] = jQuery.style( this.elem, this.prop );
		this.options.hide = true;

		// Begin the animation
		this.custom(this.cur(), 0);
	},

	// Each step of an animation
	step: function( gotoEnd ) {
		var t = fxNow || createFxNow(),
			done = true,
			elem = this.elem,
			options = this.options,
			i, n;

		if ( gotoEnd || t >= options.duration + this.startTime ) {
			this.now = this.end;
			this.pos = this.state = 1;
			this.update();

			options.animatedProperties[ this.prop ] = true;

			for ( i in options.animatedProperties ) {
				if ( options.animatedProperties[i] !== true ) {
					done = false;
				}
			}

			if ( done ) {
				// Reset the overflow
				if ( options.overflow != null && !jQuery.support.shrinkWrapBlocks ) {

					jQuery.each( [ "", "X", "Y" ], function (index, value) {
						elem.style[ "overflow" + value ] = options.overflow[index];
					});
				}

				// Hide the element if the "hide" operation was done
				if ( options.hide ) {
					jQuery(elem).hide();
				}

				// Reset the properties, if the item has been hidden or shown
				if ( options.hide || options.show ) {
					for ( var p in options.animatedProperties ) {
						jQuery.style( elem, p, options.orig[p] );
					}
				}

				// Execute the complete function
				options.complete.call( elem );
			}

			return false;

		} else {
			// classical easing cannot be used with an Infinity duration
			if ( options.duration == Infinity ) {
				this.now = t;
			} else {
				n = t - this.startTime;
				this.state = n / options.duration;

				// Perform the easing function, defaults to swing
				this.pos = jQuery.easing[ options.animatedProperties[ this.prop ] ]( this.state, n, 0, 1, options.duration );
				this.now = this.start + ((this.end - this.start) * this.pos);
			}
			// Perform the next step of the animation
			this.update();
		}

		return true;
	}
};

jQuery.extend( jQuery.fx, {
	tick: function() {
		for ( var timers = jQuery.timers, i = 0 ; i < timers.length ; ++i ) {
			if ( !timers[i]() ) {
				timers.splice(i--, 1);
			}
		}

		if ( !timers.length ) {
			jQuery.fx.stop();
		}
	},

	interval: 13,

	stop: function() {
		clearInterval( timerId );
		timerId = null;
	},

	speeds: {
		slow: 600,
		fast: 200,
		// Default speed
		_default: 400
	},

	step: {
		opacity: function( fx ) {
			jQuery.style( fx.elem, "opacity", fx.now );
		},

		_default: function( fx ) {
			if ( fx.elem.style && fx.elem.style[ fx.prop ] != null ) {
				fx.elem.style[ fx.prop ] = (fx.prop === "width" || fx.prop === "height" ? Math.max(0, fx.now) : fx.now) + fx.unit;
			} else {
				fx.elem[ fx.prop ] = fx.now;
			}
		}
	}
});

if ( jQuery.expr && jQuery.expr.filters ) {
	jQuery.expr.filters.animated = function( elem ) {
		return jQuery.grep(jQuery.timers, function( fn ) {
			return elem === fn.elem;
		}).length;
	};
}

// Try to restore the default display value of an element
function defaultDisplay( nodeName ) {

	if ( !elemdisplay[ nodeName ] ) {

		var body = document.body,
			elem = jQuery( "<" + nodeName + ">" ).appendTo( body ),
			display = elem.css( "display" );

		elem.remove();

		// If the simple way fails,
		// get element's real default display by attaching it to a temp iframe
		if ( display === "none" || display === "" ) {
			// No iframe to use yet, so create it
			if ( !iframe ) {
				iframe = document.createElement( "iframe" );
				iframe.frameBorder = iframe.width = iframe.height = 0;
			}

			body.appendChild( iframe );

			// Create a cacheable copy of the iframe document on first call.
			// IE and Opera will allow us to reuse the iframeDoc without re-writing the fake HTML
			// document to it; WebKit & Firefox won't allow reusing the iframe document.
			if ( !iframeDoc || !iframe.createElement ) {
				iframeDoc = ( iframe.contentWindow || iframe.contentDocument ).document;
				iframeDoc.write( ( document.compatMode === "CSS1Compat" ? "<!doctype html>" : "" ) + "<html><body>" );
				iframeDoc.close();
			}

			elem = iframeDoc.createElement( nodeName );

			iframeDoc.body.appendChild( elem );

			display = jQuery.css( elem, "display" );

			body.removeChild( iframe );
		}

		// Store the correct default display
		elemdisplay[ nodeName ] = display;
	}

	return elemdisplay[ nodeName ];
}




var rtable = /^t(?:able|d|h)$/i,
	rroot = /^(?:body|html)$/i;

if ( "getBoundingClientRect" in document.documentElement ) {
	jQuery.fn.offset = function( options ) {
		var elem = this[0], box;

		if ( options ) {
			return this.each(function( i ) {
				jQuery.offset.setOffset( this, options, i );
			});
		}

		if ( !elem || !elem.ownerDocument ) {
			return null;
		}

		if ( elem === elem.ownerDocument.body ) {
			return jQuery.offset.bodyOffset( elem );
		}

		try {
			box = elem.getBoundingClientRect();
		} catch(e) {}

		var doc = elem.ownerDocument,
			docElem = doc.documentElement;

		// Make sure we're not dealing with a disconnected DOM node
		if ( !box || !jQuery.contains( docElem, elem ) ) {
			return box ? { top: box.top, left: box.left } : { top: 0, left: 0 };
		}

		var body = doc.body,
			win = getWindow(doc),
			clientTop  = docElem.clientTop  || body.clientTop  || 0,
			clientLeft = docElem.clientLeft || body.clientLeft || 0,
			scrollTop  = win.pageYOffset || jQuery.support.boxModel && docElem.scrollTop  || body.scrollTop,
			scrollLeft = win.pageXOffset || jQuery.support.boxModel && docElem.scrollLeft || body.scrollLeft,
			top  = box.top  + scrollTop  - clientTop,
			left = box.left + scrollLeft - clientLeft;

		return { top: top, left: left };
	};

} else {
	jQuery.fn.offset = function( options ) {
		var elem = this[0];

		if ( options ) {
			return this.each(function( i ) {
				jQuery.offset.setOffset( this, options, i );
			});
		}

		if ( !elem || !elem.ownerDocument ) {
			return null;
		}

		if ( elem === elem.ownerDocument.body ) {
			return jQuery.offset.bodyOffset( elem );
		}

		jQuery.offset.initialize();

		var computedStyle,
			offsetParent = elem.offsetParent,
			prevOffsetParent = elem,
			doc = elem.ownerDocument,
			docElem = doc.documentElement,
			body = doc.body,
			defaultView = doc.defaultView,
			prevComputedStyle = defaultView ? defaultView.getComputedStyle( elem, null ) : elem.currentStyle,
			top = elem.offsetTop,
			left = elem.offsetLeft;

		while ( (elem = elem.parentNode) && elem !== body && elem !== docElem ) {
			if ( jQuery.offset.supportsFixedPosition && prevComputedStyle.position === "fixed" ) {
				break;
			}

			computedStyle = defaultView ? defaultView.getComputedStyle(elem, null) : elem.currentStyle;
			top  -= elem.scrollTop;
			left -= elem.scrollLeft;

			if ( elem === offsetParent ) {
				top  += elem.offsetTop;
				left += elem.offsetLeft;

				if ( jQuery.offset.doesNotAddBorder && !(jQuery.offset.doesAddBorderForTableAndCells && rtable.test(elem.nodeName)) ) {
					top  += parseFloat( computedStyle.borderTopWidth  ) || 0;
					left += parseFloat( computedStyle.borderLeftWidth ) || 0;
				}

				prevOffsetParent = offsetParent;
				offsetParent = elem.offsetParent;
			}

			if ( jQuery.offset.subtractsBorderForOverflowNotVisible && computedStyle.overflow !== "visible" ) {
				top  += parseFloat( computedStyle.borderTopWidth  ) || 0;
				left += parseFloat( computedStyle.borderLeftWidth ) || 0;
			}

			prevComputedStyle = computedStyle;
		}

		if ( prevComputedStyle.position === "relative" || prevComputedStyle.position === "static" ) {
			top  += body.offsetTop;
			left += body.offsetLeft;
		}

		if ( jQuery.offset.supportsFixedPosition && prevComputedStyle.position === "fixed" ) {
			top  += Math.max( docElem.scrollTop, body.scrollTop );
			left += Math.max( docElem.scrollLeft, body.scrollLeft );
		}

		return { top: top, left: left };
	};
}

jQuery.offset = {
	initialize: function() {
		var body = document.body, container = document.createElement("div"), innerDiv, checkDiv, table, td, bodyMarginTop = parseFloat( jQuery.css(body, "marginTop") ) || 0,
			html = "<div style='position:absolute;top:0;left:0;margin:0;border:5px solid #000;padding:0;width:1px;height:1px;'><div></div></div><table style='position:absolute;top:0;left:0;margin:0;border:5px solid #000;padding:0;width:1px;height:1px;' cellpadding='0' cellspacing='0'><tr><td></td></tr></table>";

		jQuery.extend( container.style, { position: "absolute", top: 0, left: 0, margin: 0, border: 0, width: "1px", height: "1px", visibility: "hidden" } );

		container.innerHTML = html;
		body.insertBefore( container, body.firstChild );
		innerDiv = container.firstChild;
		checkDiv = innerDiv.firstChild;
		td = innerDiv.nextSibling.firstChild.firstChild;

		this.doesNotAddBorder = (checkDiv.offsetTop !== 5);
		this.doesAddBorderForTableAndCells = (td.offsetTop === 5);

		checkDiv.style.position = "fixed";
		checkDiv.style.top = "20px";

		// safari subtracts parent border width here which is 5px
		this.supportsFixedPosition = (checkDiv.offsetTop === 20 || checkDiv.offsetTop === 15);
		checkDiv.style.position = checkDiv.style.top = "";

		innerDiv.style.overflow = "hidden";
		innerDiv.style.position = "relative";

		this.subtractsBorderForOverflowNotVisible = (checkDiv.offsetTop === -5);

		this.doesNotIncludeMarginInBodyOffset = (body.offsetTop !== bodyMarginTop);

		body.removeChild( container );
		jQuery.offset.initialize = jQuery.noop;
	},

	bodyOffset: function( body ) {
		var top = body.offsetTop,
			left = body.offsetLeft;

		jQuery.offset.initialize();

		if ( jQuery.offset.doesNotIncludeMarginInBodyOffset ) {
			top  += parseFloat( jQuery.css(body, "marginTop") ) || 0;
			left += parseFloat( jQuery.css(body, "marginLeft") ) || 0;
		}

		return { top: top, left: left };
	},

	setOffset: function( elem, options, i ) {
		var position = jQuery.css( elem, "position" );

		// set position first, in-case top/left are set even on static elem
		if ( position === "static" ) {
			elem.style.position = "relative";
		}

		var curElem = jQuery( elem ),
			curOffset = curElem.offset(),
			curCSSTop = jQuery.css( elem, "top" ),
			curCSSLeft = jQuery.css( elem, "left" ),
			calculatePosition = (position === "absolute" || position === "fixed") && jQuery.inArray("auto", [curCSSTop, curCSSLeft]) > -1,
			props = {}, curPosition = {}, curTop, curLeft;

		// need to be able to calculate position if either top or left is auto and position is either absolute or fixed
		if ( calculatePosition ) {
			curPosition = curElem.position();
			curTop = curPosition.top;
			curLeft = curPosition.left;
		} else {
			curTop = parseFloat( curCSSTop ) || 0;
			curLeft = parseFloat( curCSSLeft ) || 0;
		}

		if ( jQuery.isFunction( options ) ) {
			options = options.call( elem, i, curOffset );
		}

		if (options.top != null) {
			props.top = (options.top - curOffset.top) + curTop;
		}
		if (options.left != null) {
			props.left = (options.left - curOffset.left) + curLeft;
		}

		if ( "using" in options ) {
			options.using.call( elem, props );
		} else {
			curElem.css( props );
		}
	}
};


jQuery.fn.extend({
	position: function() {
		if ( !this[0] ) {
			return null;
		}

		var elem = this[0],

		// Get *real* offsetParent
		offsetParent = this.offsetParent(),

		// Get correct offsets
		offset       = this.offset(),
		parentOffset = rroot.test(offsetParent[0].nodeName) ? { top: 0, left: 0 } : offsetParent.offset();

		// Subtract element margins
		// note: when an element has margin: auto the offsetLeft and marginLeft
		// are the same in Safari causing offset.left to incorrectly be 0
		offset.top  -= parseFloat( jQuery.css(elem, "marginTop") ) || 0;
		offset.left -= parseFloat( jQuery.css(elem, "marginLeft") ) || 0;

		// Add offsetParent borders
		parentOffset.top  += parseFloat( jQuery.css(offsetParent[0], "borderTopWidth") ) || 0;
		parentOffset.left += parseFloat( jQuery.css(offsetParent[0], "borderLeftWidth") ) || 0;

		// Subtract the two offsets
		return {
			top:  offset.top  - parentOffset.top,
			left: offset.left - parentOffset.left
		};
	},

	offsetParent: function() {
		return this.map(function() {
			var offsetParent = this.offsetParent || document.body;
			while ( offsetParent && (!rroot.test(offsetParent.nodeName) && jQuery.css(offsetParent, "position") === "static") ) {
				offsetParent = offsetParent.offsetParent;
			}
			return offsetParent;
		});
	}
});


// Create scrollLeft and scrollTop methods
jQuery.each( ["Left", "Top"], function( i, name ) {
	var method = "scroll" + name;

	jQuery.fn[ method ] = function( val ) {
		var elem, win;

		if ( val === undefined ) {
			elem = this[ 0 ];

			if ( !elem ) {
				return null;
			}

			win = getWindow( elem );

			// Return the scroll offset
			return win ? ("pageXOffset" in win) ? win[ i ? "pageYOffset" : "pageXOffset" ] :
				jQuery.support.boxModel && win.document.documentElement[ method ] ||
					win.document.body[ method ] :
				elem[ method ];
		}

		// Set the scroll offset
		return this.each(function() {
			win = getWindow( this );

			if ( win ) {
				win.scrollTo(
					!i ? val : jQuery( win ).scrollLeft(),
					 i ? val : jQuery( win ).scrollTop()
				);

			} else {
				this[ method ] = val;
			}
		});
	};
});

function getWindow( elem ) {
	return jQuery.isWindow( elem ) ?
		elem :
		elem.nodeType === 9 ?
			elem.defaultView || elem.parentWindow :
			false;
}




// Create width, height, innerHeight, innerWidth, outerHeight and outerWidth methods
jQuery.each([ "Height", "Width" ], function( i, name ) {

	var type = name.toLowerCase();

	// innerHeight and innerWidth
	jQuery.fn[ "inner" + name ] = function() {
		var elem = this[0];
		return elem && elem.style ?
			parseFloat( jQuery.css( elem, type, "padding" ) ) :
			null;
	};

	// outerHeight and outerWidth
	jQuery.fn[ "outer" + name ] = function( margin ) {
		var elem = this[0];
		return elem && elem.style ?
			parseFloat( jQuery.css( elem, type, margin ? "margin" : "border" ) ) :
			null;
	};

	jQuery.fn[ type ] = function( size ) {
		// Get window width or height
		var elem = this[0];
		if ( !elem ) {
			return size == null ? null : this;
		}

		if ( jQuery.isFunction( size ) ) {
			return this.each(function( i ) {
				var self = jQuery( this );
				self[ type ]( size.call( this, i, self[ type ]() ) );
			});
		}

		if ( jQuery.isWindow( elem ) ) {
			// Everyone else use document.documentElement or document.body depending on Quirks vs Standards mode
			// 3rd condition allows Nokia support, as it supports the docElem prop but not CSS1Compat
			var docElemProp = elem.document.documentElement[ "client" + name ],
				body = elem.document.body;
			return elem.document.compatMode === "CSS1Compat" && docElemProp ||
				body && body[ "client" + name ] || docElemProp;

		// Get document width or height
		} else if ( elem.nodeType === 9 ) {
			// Either scroll[Width/Height] or offset[Width/Height], whichever is greater
			return Math.max(
				elem.documentElement["client" + name],
				elem.body["scroll" + name], elem.documentElement["scroll" + name],
				elem.body["offset" + name], elem.documentElement["offset" + name]
			);

		// Get or set width or height on the element
		} else if ( size === undefined ) {
			var orig = jQuery.css( elem, type ),
				ret = parseFloat( orig );

			return jQuery.isNaN( ret ) ? orig : ret;

		// Set the width or height on the element (default to pixels if value is unitless)
		} else {
			return this.css( type, typeof size === "string" ? size : size + "px" );
		}
	};

});


// Expose jQuery to the global object
window.jQuery = window.$ = jQuery;
})(window);
//     Underscore.js 1.1.7
//     (c) 2011 Jeremy Ashkenas, DocumentCloud Inc.
//     Underscore is freely distributable under the MIT license.
//     Portions of Underscore are inspired or borrowed from Prototype,
//     Oliver Steele's Functional, and John Resig's Micro-Templating.
//     For all details and documentation:
//     http://documentcloud.github.com/underscore

(function() {

  // Baseline setup
  // --------------

  // Establish the root object, `window` in the browser, or `global` on the server.
  var root = this;

  // Save the previous value of the `_` variable.
  var previousUnderscore = root._;

  // Establish the object that gets returned to break out of a loop iteration.
  var breaker = {};

  // Save bytes in the minified (but not gzipped) version:
  var ArrayProto = Array.prototype, ObjProto = Object.prototype, FuncProto = Function.prototype;

  // Create quick reference variables for speed access to core prototypes.
  var slice            = ArrayProto.slice,
      unshift          = ArrayProto.unshift,
      toString         = ObjProto.toString,
      hasOwnProperty   = ObjProto.hasOwnProperty;

  // All **ECMAScript 5** native function implementations that we hope to use
  // are declared here.
  var
    nativeForEach      = ArrayProto.forEach,
    nativeMap          = ArrayProto.map,
    nativeReduce       = ArrayProto.reduce,
    nativeReduceRight  = ArrayProto.reduceRight,
    nativeFilter       = ArrayProto.filter,
    nativeEvery        = ArrayProto.every,
    nativeSome         = ArrayProto.some,
    nativeIndexOf      = ArrayProto.indexOf,
    nativeLastIndexOf  = ArrayProto.lastIndexOf,
    nativeIsArray      = Array.isArray,
    nativeKeys         = Object.keys,
    nativeBind         = FuncProto.bind;

  // Create a safe reference to the Underscore object for use below.
  var _ = function(obj) { return new wrapper(obj); };

  // Export the Underscore object for **CommonJS**, with backwards-compatibility
  // for the old `require()` API. If we're not in CommonJS, add `_` to the
  // global object.
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = _;
    _._ = _;
  } else {
    // Exported as a string, for Closure Compiler "advanced" mode.
    root['_'] = _;
  }

  // Current version.
  _.VERSION = '1.1.7';

  // Collection Functions
  // --------------------

  // The cornerstone, an `each` implementation, aka `forEach`.
  // Handles objects with the built-in `forEach`, arrays, and raw objects.
  // Delegates to **ECMAScript 5**'s native `forEach` if available.
  var each = _.each = _.forEach = function(obj, iterator, context) {
    if (obj == null) return;
    if (nativeForEach && obj.forEach === nativeForEach) {
      obj.forEach(iterator, context);
    } else if (obj.length === +obj.length) {
      for (var i = 0, l = obj.length; i < l; i++) {
        if (i in obj && iterator.call(context, obj[i], i, obj) === breaker) return;
      }
    } else {
      for (var key in obj) {
        if (hasOwnProperty.call(obj, key)) {
          if (iterator.call(context, obj[key], key, obj) === breaker) return;
        }
      }
    }
  };

  // Return the results of applying the iterator to each element.
  // Delegates to **ECMAScript 5**'s native `map` if available.
  _.map = function(obj, iterator, context) {
    var results = [];
    if (obj == null) return results;
    if (nativeMap && obj.map === nativeMap) return obj.map(iterator, context);
    each(obj, function(value, index, list) {
      results[results.length] = iterator.call(context, value, index, list);
    });
    return results;
  };

  // **Reduce** builds up a single result from a list of values, aka `inject`,
  // or `foldl`. Delegates to **ECMAScript 5**'s native `reduce` if available.
  _.reduce = _.foldl = _.inject = function(obj, iterator, memo, context) {
    var initial = memo !== void 0;
    if (obj == null) obj = [];
    if (nativeReduce && obj.reduce === nativeReduce) {
      if (context) iterator = _.bind(iterator, context);
      return initial ? obj.reduce(iterator, memo) : obj.reduce(iterator);
    }
    each(obj, function(value, index, list) {
      if (!initial) {
        memo = value;
        initial = true;
      } else {
        memo = iterator.call(context, memo, value, index, list);
      }
    });
    if (!initial) throw new TypeError("Reduce of empty array with no initial value");
    return memo;
  };

  // The right-associative version of reduce, also known as `foldr`.
  // Delegates to **ECMAScript 5**'s native `reduceRight` if available.
  _.reduceRight = _.foldr = function(obj, iterator, memo, context) {
    if (obj == null) obj = [];
    if (nativeReduceRight && obj.reduceRight === nativeReduceRight) {
      if (context) iterator = _.bind(iterator, context);
      return memo !== void 0 ? obj.reduceRight(iterator, memo) : obj.reduceRight(iterator);
    }
    var reversed = (_.isArray(obj) ? obj.slice() : _.toArray(obj)).reverse();
    return _.reduce(reversed, iterator, memo, context);
  };

  // Return the first value which passes a truth test. Aliased as `detect`.
  _.find = _.detect = function(obj, iterator, context) {
    var result;
    any(obj, function(value, index, list) {
      if (iterator.call(context, value, index, list)) {
        result = value;
        return true;
      }
    });
    return result;
  };

  // Return all the elements that pass a truth test.
  // Delegates to **ECMAScript 5**'s native `filter` if available.
  // Aliased as `select`.
  _.filter = _.select = function(obj, iterator, context) {
    var results = [];
    if (obj == null) return results;
    if (nativeFilter && obj.filter === nativeFilter) return obj.filter(iterator, context);
    each(obj, function(value, index, list) {
      if (iterator.call(context, value, index, list)) results[results.length] = value;
    });
    return results;
  };

  // Return all the elements for which a truth test fails.
  _.reject = function(obj, iterator, context) {
    var results = [];
    if (obj == null) return results;
    each(obj, function(value, index, list) {
      if (!iterator.call(context, value, index, list)) results[results.length] = value;
    });
    return results;
  };

  // Determine whether all of the elements match a truth test.
  // Delegates to **ECMAScript 5**'s native `every` if available.
  // Aliased as `all`.
  _.every = _.all = function(obj, iterator, context) {
    var result = true;
    if (obj == null) return result;
    if (nativeEvery && obj.every === nativeEvery) return obj.every(iterator, context);
    each(obj, function(value, index, list) {
      if (!(result = result && iterator.call(context, value, index, list))) return breaker;
    });
    return result;
  };

  // Determine if at least one element in the object matches a truth test.
  // Delegates to **ECMAScript 5**'s native `some` if available.
  // Aliased as `any`.
  var any = _.some = _.any = function(obj, iterator, context) {
    iterator = iterator || _.identity;
    var result = false;
    if (obj == null) return result;
    if (nativeSome && obj.some === nativeSome) return obj.some(iterator, context);
    each(obj, function(value, index, list) {
      if (result |= iterator.call(context, value, index, list)) return breaker;
    });
    return !!result;
  };

  // Determine if a given value is included in the array or object using `===`.
  // Aliased as `contains`.
  _.include = _.contains = function(obj, target) {
    var found = false;
    if (obj == null) return found;
    if (nativeIndexOf && obj.indexOf === nativeIndexOf) return obj.indexOf(target) != -1;
    any(obj, function(value) {
      if (found = value === target) return true;
    });
    return found;
  };

  // Invoke a method (with arguments) on every item in a collection.
  _.invoke = function(obj, method) {
    var args = slice.call(arguments, 2);
    return _.map(obj, function(value) {
      return (method.call ? method || value : value[method]).apply(value, args);
    });
  };

  // Convenience version of a common use case of `map`: fetching a property.
  _.pluck = function(obj, key) {
    return _.map(obj, function(value){ return value[key]; });
  };

  // Return the maximum element or (element-based computation).
  _.max = function(obj, iterator, context) {
    if (!iterator && _.isArray(obj)) return Math.max.apply(Math, obj);
    var result = {computed : -Infinity};
    each(obj, function(value, index, list) {
      var computed = iterator ? iterator.call(context, value, index, list) : value;
      computed >= result.computed && (result = {value : value, computed : computed});
    });
    return result.value;
  };

  // Return the minimum element (or element-based computation).
  _.min = function(obj, iterator, context) {
    if (!iterator && _.isArray(obj)) return Math.min.apply(Math, obj);
    var result = {computed : Infinity};
    each(obj, function(value, index, list) {
      var computed = iterator ? iterator.call(context, value, index, list) : value;
      computed < result.computed && (result = {value : value, computed : computed});
    });
    return result.value;
  };

  // Sort the object's values by a criterion produced by an iterator.
  _.sortBy = function(obj, iterator, context) {
    return _.pluck(_.map(obj, function(value, index, list) {
      return {
        value : value,
        criteria : iterator.call(context, value, index, list)
      };
    }).sort(function(left, right) {
      var a = left.criteria, b = right.criteria;
      return a < b ? -1 : a > b ? 1 : 0;
    }), 'value');
  };

  // Groups the object's values by a criterion produced by an iterator
  _.groupBy = function(obj, iterator) {
    var result = {};
    each(obj, function(value, index) {
      var key = iterator(value, index);
      (result[key] || (result[key] = [])).push(value);
    });
    return result;
  };

  // Use a comparator function to figure out at what index an object should
  // be inserted so as to maintain order. Uses binary search.
  _.sortedIndex = function(array, obj, iterator) {
    iterator || (iterator = _.identity);
    var low = 0, high = array.length;
    while (low < high) {
      var mid = (low + high) >> 1;
      iterator(array[mid]) < iterator(obj) ? low = mid + 1 : high = mid;
    }
    return low;
  };

  // Safely convert anything iterable into a real, live array.
  _.toArray = function(iterable) {
    if (!iterable)                return [];
    if (iterable.toArray)         return iterable.toArray();
    if (_.isArray(iterable))      return slice.call(iterable);
    if (_.isArguments(iterable))  return slice.call(iterable);
    return _.values(iterable);
  };

  // Return the number of elements in an object.
  _.size = function(obj) {
    return _.toArray(obj).length;
  };

  // Array Functions
  // ---------------

  // Get the first element of an array. Passing **n** will return the first N
  // values in the array. Aliased as `head`. The **guard** check allows it to work
  // with `_.map`.
  _.first = _.head = function(array, n, guard) {
    return (n != null) && !guard ? slice.call(array, 0, n) : array[0];
  };

  // Returns everything but the first entry of the array. Aliased as `tail`.
  // Especially useful on the arguments object. Passing an **index** will return
  // the rest of the values in the array from that index onward. The **guard**
  // check allows it to work with `_.map`.
  _.rest = _.tail = function(array, index, guard) {
    return slice.call(array, (index == null) || guard ? 1 : index);
  };

  // Get the last element of an array.
  _.last = function(array) {
    return array[array.length - 1];
  };

  // Trim out all falsy values from an array.
  _.compact = function(array) {
    return _.filter(array, function(value){ return !!value; });
  };

  // Return a completely flattened version of an array.
  _.flatten = function(array) {
    return _.reduce(array, function(memo, value) {
      if (_.isArray(value)) return memo.concat(_.flatten(value));
      memo[memo.length] = value;
      return memo;
    }, []);
  };

  // Return a version of the array that does not contain the specified value(s).
  _.without = function(array) {
    return _.difference(array, slice.call(arguments, 1));
  };

  // Produce a duplicate-free version of the array. If the array has already
  // been sorted, you have the option of using a faster algorithm.
  // Aliased as `unique`.
  _.uniq = _.unique = function(array, isSorted) {
    return _.reduce(array, function(memo, el, i) {
      if (0 == i || (isSorted === true ? _.last(memo) != el : !_.include(memo, el))) memo[memo.length] = el;
      return memo;
    }, []);
  };

  // Produce an array that contains the union: each distinct element from all of
  // the passed-in arrays.
  _.union = function() {
    return _.uniq(_.flatten(arguments));
  };

  // Produce an array that contains every item shared between all the
  // passed-in arrays. (Aliased as "intersect" for back-compat.)
  _.intersection = _.intersect = function(array) {
    var rest = slice.call(arguments, 1);
    return _.filter(_.uniq(array), function(item) {
      return _.every(rest, function(other) {
        return _.indexOf(other, item) >= 0;
      });
    });
  };

  // Take the difference between one array and another.
  // Only the elements present in just the first array will remain.
  _.difference = function(array, other) {
    return _.filter(array, function(value){ return !_.include(other, value); });
  };

  // Zip together multiple lists into a single array -- elements that share
  // an index go together.
  _.zip = function() {
    var args = slice.call(arguments);
    var length = _.max(_.pluck(args, 'length'));
    var results = new Array(length);
    for (var i = 0; i < length; i++) results[i] = _.pluck(args, "" + i);
    return results;
  };

  // If the browser doesn't supply us with indexOf (I'm looking at you, **MSIE**),
  // we need this function. Return the position of the first occurrence of an
  // item in an array, or -1 if the item is not included in the array.
  // Delegates to **ECMAScript 5**'s native `indexOf` if available.
  // If the array is large and already in sort order, pass `true`
  // for **isSorted** to use binary search.
  _.indexOf = function(array, item, isSorted) {
    if (array == null) return -1;
    var i, l;
    if (isSorted) {
      i = _.sortedIndex(array, item);
      return array[i] === item ? i : -1;
    }
    if (nativeIndexOf && array.indexOf === nativeIndexOf) return array.indexOf(item);
    for (i = 0, l = array.length; i < l; i++) if (array[i] === item) return i;
    return -1;
  };


  // Delegates to **ECMAScript 5**'s native `lastIndexOf` if available.
  _.lastIndexOf = function(array, item) {
    if (array == null) return -1;
    if (nativeLastIndexOf && array.lastIndexOf === nativeLastIndexOf) return array.lastIndexOf(item);
    var i = array.length;
    while (i--) if (array[i] === item) return i;
    return -1;
  };

  // Generate an integer Array containing an arithmetic progression. A port of
  // the native Python `range()` function. See
  // [the Python documentation](http://docs.python.org/library/functions.html#range).
  _.range = function(start, stop, step) {
    if (arguments.length <= 1) {
      stop = start || 0;
      start = 0;
    }
    step = arguments[2] || 1;

    var len = Math.max(Math.ceil((stop - start) / step), 0);
    var idx = 0;
    var range = new Array(len);

    while(idx < len) {
      range[idx++] = start;
      start += step;
    }

    return range;
  };

  // Function (ahem) Functions
  // ------------------

  // Create a function bound to a given object (assigning `this`, and arguments,
  // optionally). Binding with arguments is also known as `curry`.
  // Delegates to **ECMAScript 5**'s native `Function.bind` if available.
  // We check for `func.bind` first, to fail fast when `func` is undefined.
  _.bind = function(func, obj) {
    if (func.bind === nativeBind && nativeBind) return nativeBind.apply(func, slice.call(arguments, 1));
    var args = slice.call(arguments, 2);
    return function() {
      return func.apply(obj, args.concat(slice.call(arguments)));
    };
  };

  // Bind all of an object's methods to that object. Useful for ensuring that
  // all callbacks defined on an object belong to it.
  _.bindAll = function(obj) {
    var funcs = slice.call(arguments, 1);
    if (funcs.length == 0) funcs = _.functions(obj);
    each(funcs, function(f) { obj[f] = _.bind(obj[f], obj); });
    return obj;
  };

  // Memoize an expensive function by storing its results.
  _.memoize = function(func, hasher) {
    var memo = {};
    hasher || (hasher = _.identity);
    return function() {
      var key = hasher.apply(this, arguments);
      return hasOwnProperty.call(memo, key) ? memo[key] : (memo[key] = func.apply(this, arguments));
    };
  };

  // Delays a function for the given number of milliseconds, and then calls
  // it with the arguments supplied.
  _.delay = function(func, wait) {
    var args = slice.call(arguments, 2);
    return setTimeout(function(){ return func.apply(func, args); }, wait);
  };

  // Defers a function, scheduling it to run after the current call stack has
  // cleared.
  _.defer = function(func) {
    return _.delay.apply(_, [func, 1].concat(slice.call(arguments, 1)));
  };

  // Internal function used to implement `_.throttle` and `_.debounce`.
  var limit = function(func, wait, debounce) {
    var timeout;
    return function() {
      var context = this, args = arguments;
      var throttler = function() {
        timeout = null;
        func.apply(context, args);
      };
      if (debounce) clearTimeout(timeout);
      if (debounce || !timeout) timeout = setTimeout(throttler, wait);
    };
  };

  // Returns a function, that, when invoked, will only be triggered at most once
  // during a given window of time.
  _.throttle = function(func, wait) {
    return limit(func, wait, false);
  };

  // Returns a function, that, as long as it continues to be invoked, will not
  // be triggered. The function will be called after it stops being called for
  // N milliseconds.
  _.debounce = function(func, wait) {
    return limit(func, wait, true);
  };

  // Returns a function that will be executed at most one time, no matter how
  // often you call it. Useful for lazy initialization.
  _.once = function(func) {
    var ran = false, memo;
    return function() {
      if (ran) return memo;
      ran = true;
      return memo = func.apply(this, arguments);
    };
  };

  // Returns the first function passed as an argument to the second,
  // allowing you to adjust arguments, run code before and after, and
  // conditionally execute the original function.
  _.wrap = function(func, wrapper) {
    return function() {
      var args = [func].concat(slice.call(arguments));
      return wrapper.apply(this, args);
    };
  };

  // Returns a function that is the composition of a list of functions, each
  // consuming the return value of the function that follows.
  _.compose = function() {
    var funcs = slice.call(arguments);
    return function() {
      var args = slice.call(arguments);
      for (var i = funcs.length - 1; i >= 0; i--) {
        args = [funcs[i].apply(this, args)];
      }
      return args[0];
    };
  };

  // Returns a function that will only be executed after being called N times.
  _.after = function(times, func) {
    return function() {
      if (--times < 1) { return func.apply(this, arguments); }
    };
  };


  // Object Functions
  // ----------------

  // Retrieve the names of an object's properties.
  // Delegates to **ECMAScript 5**'s native `Object.keys`
  _.keys = nativeKeys || function(obj) {
    if (obj !== Object(obj)) throw new TypeError('Invalid object');
    var keys = [];
    for (var key in obj) if (hasOwnProperty.call(obj, key)) keys[keys.length] = key;
    return keys;
  };

  // Retrieve the values of an object's properties.
  _.values = function(obj) {
    return _.map(obj, _.identity);
  };

  // Return a sorted list of the function names available on the object.
  // Aliased as `methods`
  _.functions = _.methods = function(obj) {
    var names = [];
    for (var key in obj) {
      if (_.isFunction(obj[key])) names.push(key);
    }
    return names.sort();
  };

  // Extend a given object with all the properties in passed-in object(s).
  _.extend = function(obj) {
    each(slice.call(arguments, 1), function(source) {
      for (var prop in source) {
        if (source[prop] !== void 0) obj[prop] = source[prop];
      }
    });
    return obj;
  };

  // Fill in a given object with default properties.
  _.defaults = function(obj) {
    each(slice.call(arguments, 1), function(source) {
      for (var prop in source) {
        if (obj[prop] == null) obj[prop] = source[prop];
      }
    });
    return obj;
  };

  // Create a (shallow-cloned) duplicate of an object.
  _.clone = function(obj) {
    return _.isArray(obj) ? obj.slice() : _.extend({}, obj);
  };

  // Invokes interceptor with the obj, and then returns obj.
  // The primary purpose of this method is to "tap into" a method chain, in
  // order to perform operations on intermediate results within the chain.
  _.tap = function(obj, interceptor) {
    interceptor(obj);
    return obj;
  };

  // Perform a deep comparison to check if two objects are equal.
  _.isEqual = function(a, b) {
    // Check object identity.
    if (a === b) return true;
    // Different types?
    var atype = typeof(a), btype = typeof(b);
    if (atype != btype) return false;
    // Basic equality test (watch out for coercions).
    if (a == b) return true;
    // One is falsy and the other truthy.
    if ((!a && b) || (a && !b)) return false;
    // Unwrap any wrapped objects.
    if (a._chain) a = a._wrapped;
    if (b._chain) b = b._wrapped;
    // One of them implements an isEqual()?
    if (a.isEqual) return a.isEqual(b);
    if (b.isEqual) return b.isEqual(a);
    // Check dates' integer values.
    if (_.isDate(a) && _.isDate(b)) return a.getTime() === b.getTime();
    // Both are NaN?
    if (_.isNaN(a) && _.isNaN(b)) return false;
    // Compare regular expressions.
    if (_.isRegExp(a) && _.isRegExp(b))
      return a.source     === b.source &&
             a.global     === b.global &&
             a.ignoreCase === b.ignoreCase &&
             a.multiline  === b.multiline;
    // If a is not an object by this point, we can't handle it.
    if (atype !== 'object') return false;
    // Check for different array lengths before comparing contents.
    if (a.length && (a.length !== b.length)) return false;
    // Nothing else worked, deep compare the contents.
    var aKeys = _.keys(a), bKeys = _.keys(b);
    // Different object sizes?
    if (aKeys.length != bKeys.length) return false;
    // Recursive comparison of contents.
    for (var key in a) if (!(key in b) || !_.isEqual(a[key], b[key])) return false;
    return true;
  };

  // Is a given array or object empty?
  _.isEmpty = function(obj) {
    if (_.isArray(obj) || _.isString(obj)) return obj.length === 0;
    for (var key in obj) if (hasOwnProperty.call(obj, key)) return false;
    return true;
  };

  // Is a given value a DOM element?
  _.isElement = function(obj) {
    return !!(obj && obj.nodeType == 1);
  };

  // Is a given value an array?
  // Delegates to ECMA5's native Array.isArray
  _.isArray = nativeIsArray || function(obj) {
    return toString.call(obj) === '[object Array]';
  };

  // Is a given variable an object?
  _.isObject = function(obj) {
    return obj === Object(obj);
  };

  // Is a given variable an arguments object?
  _.isArguments = function(obj) {
    return !!(obj && hasOwnProperty.call(obj, 'callee'));
  };

  // Is a given value a function?
  _.isFunction = function(obj) {
    return !!(obj && obj.constructor && obj.call && obj.apply);
  };

  // Is a given value a string?
  _.isString = function(obj) {
    return !!(obj === '' || (obj && obj.charCodeAt && obj.substr));
  };

  // Is a given value a number?
  _.isNumber = function(obj) {
    return !!(obj === 0 || (obj && obj.toExponential && obj.toFixed));
  };

  // Is the given value `NaN`? `NaN` happens to be the only value in JavaScript
  // that does not equal itself.
  _.isNaN = function(obj) {
    return obj !== obj;
  };

  // Is a given value a boolean?
  _.isBoolean = function(obj) {
    return obj === true || obj === false;
  };

  // Is a given value a date?
  _.isDate = function(obj) {
    return !!(obj && obj.getTimezoneOffset && obj.setUTCFullYear);
  };

  // Is the given value a regular expression?
  _.isRegExp = function(obj) {
    return !!(obj && obj.test && obj.exec && (obj.ignoreCase || obj.ignoreCase === false));
  };

  // Is a given value equal to null?
  _.isNull = function(obj) {
    return obj === null;
  };

  // Is a given variable undefined?
  _.isUndefined = function(obj) {
    return obj === void 0;
  };

  // Utility Functions
  // -----------------

  // Run Underscore.js in *noConflict* mode, returning the `_` variable to its
  // previous owner. Returns a reference to the Underscore object.
  _.noConflict = function() {
    root._ = previousUnderscore;
    return this;
  };

  // Keep the identity function around for default iterators.
  _.identity = function(value) {
    return value;
  };

  // Run a function **n** times.
  _.times = function (n, iterator, context) {
    for (var i = 0; i < n; i++) iterator.call(context, i);
  };

  // Add your own custom functions to the Underscore object, ensuring that
  // they're correctly added to the OOP wrapper as well.
  _.mixin = function(obj) {
    each(_.functions(obj), function(name){
      addToWrapper(name, _[name] = obj[name]);
    });
  };

  // Generate a unique integer id (unique within the entire client session).
  // Useful for temporary DOM ids.
  var idCounter = 0;
  _.uniqueId = function(prefix) {
    var id = idCounter++;
    return prefix ? prefix + id : id;
  };

  // By default, Underscore uses ERB-style template delimiters, change the
  // following template settings to use alternative delimiters.
  _.templateSettings = {
    evaluate    : /<%([\s\S]+?)%>/g,
    interpolate : /<%=([\s\S]+?)%>/g
  };

  // JavaScript micro-templating, similar to John Resig's implementation.
  // Underscore templating handles arbitrary delimiters, preserves whitespace,
  // and correctly escapes quotes within interpolated code.
  _.template = function(str, data) {
    var c  = _.templateSettings;
    var tmpl = 'var __p=[],print=function(){__p.push.apply(__p,arguments);};' +
      'with(obj||{}){__p.push(\'' +
      str.replace(/\\/g, '\\\\')
         .replace(/'/g, "\\'")
         .replace(c.interpolate, function(match, code) {
           return "'," + code.replace(/\\'/g, "'") + ",'";
         })
         .replace(c.evaluate || null, function(match, code) {
           return "');" + code.replace(/\\'/g, "'")
                              .replace(/[\r\n\t]/g, ' ') + "__p.push('";
         })
         .replace(/\r/g, '\\r')
         .replace(/\n/g, '\\n')
         .replace(/\t/g, '\\t')
         + "');}return __p.join('');";
    var func = new Function('obj', tmpl);
    return data ? func(data) : func;
  };

  // The OOP Wrapper
  // ---------------

  // If Underscore is called as a function, it returns a wrapped object that
  // can be used OO-style. This wrapper holds altered versions of all the
  // underscore functions. Wrapped objects may be chained.
  var wrapper = function(obj) { this._wrapped = obj; };

  // Expose `wrapper.prototype` as `_.prototype`
  _.prototype = wrapper.prototype;

  // Helper function to continue chaining intermediate results.
  var result = function(obj, chain) {
    return chain ? _(obj).chain() : obj;
  };

  // A method to easily add functions to the OOP wrapper.
  var addToWrapper = function(name, func) {
    wrapper.prototype[name] = function() {
      var args = slice.call(arguments);
      unshift.call(args, this._wrapped);
      return result(func.apply(_, args), this._chain);
    };
  };

  // Add all of the Underscore functions to the wrapper object.
  _.mixin(_);

  // Add all mutator Array functions to the wrapper.
  each(['pop', 'push', 'reverse', 'shift', 'sort', 'splice', 'unshift'], function(name) {
    var method = ArrayProto[name];
    wrapper.prototype[name] = function() {
      method.apply(this._wrapped, arguments);
      return result(this._wrapped, this._chain);
    };
  });

  // Add all accessor Array functions to the wrapper.
  each(['concat', 'join', 'slice'], function(name) {
    var method = ArrayProto[name];
    wrapper.prototype[name] = function() {
      return result(method.apply(this._wrapped, arguments), this._chain);
    };
  });

  // Start chaining a wrapped Underscore object.
  wrapper.prototype.chain = function() {
    this._chain = true;
    return this;
  };

  // Extracts the result from a wrapped and chained object.
  wrapper.prototype.value = function() {
    return this._wrapped;
  };

})();
//     Backbone.js 0.5.3
//     (c) 2010 Jeremy Ashkenas, DocumentCloud Inc.
//     Backbone may be freely distributed under the MIT license.
//     For all details and documentation:
//     http://documentcloud.github.com/backbone

(function(){

  // Initial Setup
  // -------------

  // Save a reference to the global object.
  var root = this;

  // Save the previous value of the `Backbone` variable.
  var previousBackbone = root.Backbone;

  // The top-level namespace. All public Backbone classes and modules will
  // be attached to this. Exported for both CommonJS and the browser.
  var Backbone;
  if (typeof exports !== 'undefined') {
    Backbone = exports;
  } else {
    Backbone = root.Backbone = {};
  }

  // Current version of the library. Keep in sync with `package.json`.
  Backbone.VERSION = '0.5.3';

  // Require Underscore, if we're on the server, and it's not already present.
  var _ = root._;
  if (!_ && (typeof require !== 'undefined')) _ = require('underscore')._;

  // For Backbone's purposes, jQuery or Zepto owns the `$` variable.
  var $ = root.jQuery || root.Zepto;

  // Runs Backbone.js in *noConflict* mode, returning the `Backbone` variable
  // to its previous owner. Returns a reference to this Backbone object.
  Backbone.noConflict = function() {
    root.Backbone = previousBackbone;
    return this;
  };

  // Turn on `emulateHTTP` to support legacy HTTP servers. Setting this option will
  // fake `"PUT"` and `"DELETE"` requests via the `_method` parameter and set a
  // `X-Http-Method-Override` header.
  Backbone.emulateHTTP = false;

  // Turn on `emulateJSON` to support legacy servers that can't deal with direct
  // `application/json` requests ... will encode the body as
  // `application/x-www-form-urlencoded` instead and will send the model in a
  // form param named `model`.
  Backbone.emulateJSON = false;

  // Backbone.Events
  // -----------------

  // A module that can be mixed in to *any object* in order to provide it with
  // custom events. You may `bind` or `unbind` a callback function to an event;
  // `trigger`-ing an event fires all callbacks in succession.
  //
  //     var object = {};
  //     _.extend(object, Backbone.Events);
  //     object.bind('expand', function(){ alert('expanded'); });
  //     object.trigger('expand');
  //
  Backbone.Events = {

    // Bind an event, specified by a string name, `ev`, to a `callback` function.
    // Passing `"all"` will bind the callback to all events fired.
    bind : function(ev, callback, context) {
      var calls = this._callbacks || (this._callbacks = {});
      var list  = calls[ev] || (calls[ev] = []);
      list.push([callback, context]);
      return this;
    },

    // Remove one or many callbacks. If `callback` is null, removes all
    // callbacks for the event. If `ev` is null, removes all bound callbacks
    // for all events.
    unbind : function(ev, callback) {
      var calls;
      if (!ev) {
        this._callbacks = {};
      } else if (calls = this._callbacks) {
        if (!callback) {
          calls[ev] = [];
        } else {
          var list = calls[ev];
          if (!list) return this;
          for (var i = 0, l = list.length; i < l; i++) {
            if (list[i] && callback === list[i][0]) {
              list[i] = null;
              break;
            }
          }
        }
      }
      return this;
    },

    // Trigger an event, firing all bound callbacks. Callbacks are passed the
    // same arguments as `trigger` is, apart from the event name.
    // Listening for `"all"` passes the true event name as the first argument.
    trigger : function(eventName) {
      var list, calls, ev, callback, args;
      var both = 2;
      if (!(calls = this._callbacks)) return this;
      while (both--) {
        ev = both ? eventName : 'all';
        if (list = calls[ev]) {
          for (var i = 0, l = list.length; i < l; i++) {
            if (!(callback = list[i])) {
              list.splice(i, 1); i--; l--;
            } else {
              args = both ? Array.prototype.slice.call(arguments, 1) : arguments;
              callback[0].apply(callback[1] || this, args);
            }
          }
        }
      }
      return this;
    }

  };

  // Backbone.Model
  // --------------

  // Create a new model, with defined attributes. A client id (`cid`)
  // is automatically generated and assigned for you.
  Backbone.Model = function(attributes, options) {
    var defaults;
    attributes || (attributes = {});
    if (defaults = this.defaults) {
      if (_.isFunction(defaults)) defaults = defaults.call(this);
      attributes = _.extend({}, defaults, attributes);
    }
    this.attributes = {};
    this._escapedAttributes = {};
    this.cid = _.uniqueId('c');
    this.set(attributes, {silent : true});
    this._changed = false;
    this._previousAttributes = _.clone(this.attributes);
    if (options && options.collection) this.collection = options.collection;
    this.initialize(attributes, options);
  };

  // Attach all inheritable methods to the Model prototype.
  _.extend(Backbone.Model.prototype, Backbone.Events, {

    // A snapshot of the model's previous attributes, taken immediately
    // after the last `"change"` event was fired.
    _previousAttributes : null,

    // Has the item been changed since the last `"change"` event?
    _changed : false,

    // The default name for the JSON `id` attribute is `"id"`. MongoDB and
    // CouchDB users may want to set this to `"_id"`.
    idAttribute : 'id',

    // Initialize is an empty function by default. Override it with your own
    // initialization logic.
    initialize : function(){},

    // Return a copy of the model's `attributes` object.
    toJSON : function() {
      return _.clone(this.attributes);
    },

    // Get the value of an attribute.
    get : function(attr) {
      return this.attributes[attr];
    },

    // Get the HTML-escaped value of an attribute.
    escape : function(attr) {
      var html;
      if (html = this._escapedAttributes[attr]) return html;
      var val = this.attributes[attr];
      return this._escapedAttributes[attr] = escapeHTML(val == null ? '' : '' + val);
    },

    // Returns `true` if the attribute contains a value that is not null
    // or undefined.
    has : function(attr) {
      return this.attributes[attr] != null;
    },

    // Set a hash of model attributes on the object, firing `"change"` unless you
    // choose to silence it.
    set : function(attrs, options) {

      // Extract attributes and options.
      options || (options = {});
      if (!attrs) return this;
      if (attrs.attributes) attrs = attrs.attributes;
      var now = this.attributes, escaped = this._escapedAttributes;

      // Run validation.
      if (!options.silent && this.validate && !this._performValidation(attrs, options)) return false;

      // Check for changes of `id`.
      if (this.idAttribute in attrs) this.id = attrs[this.idAttribute];

      // We're about to start triggering change events.
      var alreadyChanging = this._changing;
      this._changing = true;

      // Update attributes.
      for (var attr in attrs) {
        var val = attrs[attr];
        if (!_.isEqual(now[attr], val)) {
          now[attr] = val;
          delete escaped[attr];
          this._changed = true;
          if (!options.silent) this.trigger('change:' + attr, this, val, options);
        }
      }

      // Fire the `"change"` event, if the model has been changed.
      if (!alreadyChanging && !options.silent && this._changed) this.change(options);
      this._changing = false;
      return this;
    },

    // Remove an attribute from the model, firing `"change"` unless you choose
    // to silence it. `unset` is a noop if the attribute doesn't exist.
    unset : function(attr, options) {
      if (!(attr in this.attributes)) return this;
      options || (options = {});
      var value = this.attributes[attr];

      // Run validation.
      var validObj = {};
      validObj[attr] = void 0;
      if (!options.silent && this.validate && !this._performValidation(validObj, options)) return false;

      // Remove the attribute.
      delete this.attributes[attr];
      delete this._escapedAttributes[attr];
      if (attr == this.idAttribute) delete this.id;
      this._changed = true;
      if (!options.silent) {
        this.trigger('change:' + attr, this, void 0, options);
        this.change(options);
      }
      return this;
    },

    // Clear all attributes on the model, firing `"change"` unless you choose
    // to silence it.
    clear : function(options) {
      options || (options = {});
      var attr;
      var old = this.attributes;

      // Run validation.
      var validObj = {};
      for (attr in old) validObj[attr] = void 0;
      if (!options.silent && this.validate && !this._performValidation(validObj, options)) return false;

      this.attributes = {};
      this._escapedAttributes = {};
      this._changed = true;
      if (!options.silent) {
        for (attr in old) {
          this.trigger('change:' + attr, this, void 0, options);
        }
        this.change(options);
      }
      return this;
    },

    // Fetch the model from the server. If the server's representation of the
    // model differs from its current attributes, they will be overriden,
    // triggering a `"change"` event.
    fetch : function(options) {
      options || (options = {});
      var model = this;
      var success = options.success;
      options.success = function(resp, status, xhr) {
        if (!model.set(model.parse(resp, xhr), options)) return false;
        if (success) success(model, resp);
      };
      options.error = wrapError(options.error, model, options);
      return (this.sync || Backbone.sync).call(this, 'read', this, options);
    },

    // Set a hash of model attributes, and sync the model to the server.
    // If the server returns an attributes hash that differs, the model's
    // state will be `set` again.
    save : function(attrs, options) {
      options || (options = {});
      if (attrs && !this.set(attrs, options)) return false;
      var model = this;
      var success = options.success;
      options.success = function(resp, status, xhr) {
        if (!model.set(model.parse(resp, xhr), options)) return false;
        if (success) success(model, resp, xhr);
      };
      options.error = wrapError(options.error, model, options);
      var method = this.isNew() ? 'create' : 'update';
      return (this.sync || Backbone.sync).call(this, method, this, options);
    },

    // Destroy this model on the server if it was already persisted. Upon success, the model is removed
    // from its collection, if it has one.
    destroy : function(options) {
      options || (options = {});
      if (this.isNew()) return this.trigger('destroy', this, this.collection, options);
      var model = this;
      var success = options.success;
      options.success = function(resp) {
        model.trigger('destroy', model, model.collection, options);
        if (success) success(model, resp);
      };
      options.error = wrapError(options.error, model, options);
      return (this.sync || Backbone.sync).call(this, 'delete', this, options);
    },

    // Default URL for the model's representation on the server -- if you're
    // using Backbone's restful methods, override this to change the endpoint
    // that will be called.
    url : function() {
      var base = getUrl(this.collection) || this.urlRoot || urlError();
      if (this.isNew()) return base;
      return base + (base.charAt(base.length - 1) == '/' ? '' : '/') + encodeURIComponent(this.id);
    },

    // **parse** converts a response into the hash of attributes to be `set` on
    // the model. The default implementation is just to pass the response along.
    parse : function(resp, xhr) {
      return resp;
    },

    // Create a new model with identical attributes to this one.
    clone : function() {
      return new this.constructor(this);
    },

    // A model is new if it has never been saved to the server, and lacks an id.
    isNew : function() {
      return this.id == null;
    },

    // Call this method to manually fire a `change` event for this model.
    // Calling this will cause all objects observing the model to update.
    change : function(options) {
      this.trigger('change', this, options);
      this._previousAttributes = _.clone(this.attributes);
      this._changed = false;
    },

    // Determine if the model has changed since the last `"change"` event.
    // If you specify an attribute name, determine if that attribute has changed.
    hasChanged : function(attr) {
      if (attr) return this._previousAttributes[attr] != this.attributes[attr];
      return this._changed;
    },

    // Return an object containing all the attributes that have changed, or false
    // if there are no changed attributes. Useful for determining what parts of a
    // view need to be updated and/or what attributes need to be persisted to
    // the server.
    changedAttributes : function(now) {
      now || (now = this.attributes);
      var old = this._previousAttributes;
      var changed = false;
      for (var attr in now) {
        if (!_.isEqual(old[attr], now[attr])) {
          changed = changed || {};
          changed[attr] = now[attr];
        }
      }
      return changed;
    },

    // Get the previous value of an attribute, recorded at the time the last
    // `"change"` event was fired.
    previous : function(attr) {
      if (!attr || !this._previousAttributes) return null;
      return this._previousAttributes[attr];
    },

    // Get all of the attributes of the model at the time of the previous
    // `"change"` event.
    previousAttributes : function() {
      return _.clone(this._previousAttributes);
    },

    // Run validation against a set of incoming attributes, returning `true`
    // if all is well. If a specific `error` callback has been passed,
    // call that instead of firing the general `"error"` event.
    _performValidation : function(attrs, options) {
      var error = this.validate(attrs);
      if (error) {
        if (options.error) {
          options.error(this, error, options);
        } else {
          this.trigger('error', this, error, options);
        }
        return false;
      }
      return true;
    }

  });

  // Backbone.Collection
  // -------------------

  // Provides a standard collection class for our sets of models, ordered
  // or unordered. If a `comparator` is specified, the Collection will maintain
  // its models in sort order, as they're added and removed.
  Backbone.Collection = function(models, options) {
    options || (options = {});
    if (options.comparator) this.comparator = options.comparator;
    _.bindAll(this, '_onModelEvent', '_removeReference');
    this._reset();
    if (models) this.reset(models, {silent: true});
    this.initialize.apply(this, arguments);
  };

  // Define the Collection's inheritable methods.
  _.extend(Backbone.Collection.prototype, Backbone.Events, {

    // The default model for a collection is just a **Backbone.Model**.
    // This should be overridden in most cases.
    model : Backbone.Model,

    // Initialize is an empty function by default. Override it with your own
    // initialization logic.
    initialize : function(){},

    // The JSON representation of a Collection is an array of the
    // models' attributes.
    toJSON : function() {
      return this.map(function(model){ return model.toJSON(); });
    },

    // Add a model, or list of models to the set. Pass **silent** to avoid
    // firing the `added` event for every new model.
    add : function(models, options) {
      if (_.isArray(models)) {
        for (var i = 0, l = models.length; i < l; i++) {
          this._add(models[i], options);
        }
      } else {
        this._add(models, options);
      }
      return this;
    },

    // Remove a model, or a list of models from the set. Pass silent to avoid
    // firing the `removed` event for every model removed.
    remove : function(models, options) {
      if (_.isArray(models)) {
        for (var i = 0, l = models.length; i < l; i++) {
          this._remove(models[i], options);
        }
      } else {
        this._remove(models, options);
      }
      return this;
    },

    // Get a model from the set by id.
    get : function(id) {
      if (id == null) return null;
      return this._byId[id.id != null ? id.id : id];
    },

    // Get a model from the set by client id.
    getByCid : function(cid) {
      return cid && this._byCid[cid.cid || cid];
    },

    // Get the model at the given index.
    at: function(index) {
      return this.models[index];
    },

    // Force the collection to re-sort itself. You don't need to call this under normal
    // circumstances, as the set will maintain sort order as each item is added.
    sort : function(options) {
      options || (options = {});
      if (!this.comparator) throw new Error('Cannot sort a set without a comparator');
      this.models = this.sortBy(this.comparator);
      if (!options.silent) this.trigger('reset', this, options);
      return this;
    },

    // Pluck an attribute from each model in the collection.
    pluck : function(attr) {
      return _.map(this.models, function(model){ return model.get(attr); });
    },

    // When you have more items than you want to add or remove individually,
    // you can reset the entire set with a new list of models, without firing
    // any `added` or `removed` events. Fires `reset` when finished.
    reset : function(models, options) {
      models  || (models = []);
      options || (options = {});
      this.each(this._removeReference);
      this._reset();
      this.add(models, {silent: true});
      if (!options.silent) this.trigger('reset', this, options);
      return this;
    },

    // Fetch the default set of models for this collection, resetting the
    // collection when they arrive. If `add: true` is passed, appends the
    // models to the collection instead of resetting.
    fetch : function(options) {
      options || (options = {});
      var collection = this;
      var success = options.success;
      options.success = function(resp, status, xhr) {
        collection[options.add ? 'add' : 'reset'](collection.parse(resp, xhr), options);
        if (success) success(collection, resp);
      };
      options.error = wrapError(options.error, collection, options);
      return (this.sync || Backbone.sync).call(this, 'read', this, options);
    },

    // Create a new instance of a model in this collection. After the model
    // has been created on the server, it will be added to the collection.
    // Returns the model, or 'false' if validation on a new model fails.
    create : function(model, options) {
      var coll = this;
      options || (options = {});
      model = this._prepareModel(model, options);
      if (!model) return false;
      var success = options.success;
      options.success = function(nextModel, resp, xhr) {
        coll.add(nextModel, options);
        if (success) success(nextModel, resp, xhr);
      };
      model.save(null, options);
      return model;
    },

    // **parse** converts a response into a list of models to be added to the
    // collection. The default implementation is just to pass it through.
    parse : function(resp, xhr) {
      return resp;
    },

    // Proxy to _'s chain. Can't be proxied the same way the rest of the
    // underscore methods are proxied because it relies on the underscore
    // constructor.
    chain: function () {
      return _(this.models).chain();
    },

    // Reset all internal state. Called when the collection is reset.
    _reset : function(options) {
      this.length = 0;
      this.models = [];
      this._byId  = {};
      this._byCid = {};
    },

    // Prepare a model to be added to this collection
    _prepareModel: function(model, options) {
      if (!(model instanceof Backbone.Model)) {
        var attrs = model;
        model = new this.model(attrs, {collection: this});
        if (model.validate && !model._performValidation(attrs, options)) model = false;
      } else if (!model.collection) {
        model.collection = this;
      }
      return model;
    },

    // Internal implementation of adding a single model to the set, updating
    // hash indexes for `id` and `cid` lookups.
    // Returns the model, or 'false' if validation on a new model fails.
    _add : function(model, options) {
      options || (options = {});
      model = this._prepareModel(model, options);
      if (!model) return false;
      var already = this.getByCid(model);
      if (already) throw new Error(["Can't add the same model to a set twice", already.id]);
      this._byId[model.id] = model;
      this._byCid[model.cid] = model;
      var index = options.at != null ? options.at :
                  this.comparator ? this.sortedIndex(model, this.comparator) :
                  this.length;
      this.models.splice(index, 0, model);
      model.bind('all', this._onModelEvent);
      this.length++;
      if (!options.silent) model.trigger('add', model, this, options);
      return model;
    },

    // Internal implementation of removing a single model from the set, updating
    // hash indexes for `id` and `cid` lookups.
    _remove : function(model, options) {
      options || (options = {});
      model = this.getByCid(model) || this.get(model);
      if (!model) return null;
      delete this._byId[model.id];
      delete this._byCid[model.cid];
      this.models.splice(this.indexOf(model), 1);
      this.length--;
      if (!options.silent) model.trigger('remove', model, this, options);
      this._removeReference(model);
      return model;
    },

    // Internal method to remove a model's ties to a collection.
    _removeReference : function(model) {
      if (this == model.collection) {
        delete model.collection;
      }
      model.unbind('all', this._onModelEvent);
    },

    // Internal method called every time a model in the set fires an event.
    // Sets need to update their indexes when models change ids. All other
    // events simply proxy through. "add" and "remove" events that originate
    // in other collections are ignored.
    _onModelEvent : function(ev, model, collection, options) {
      if ((ev == 'add' || ev == 'remove') && collection != this) return;
      if (ev == 'destroy') {
        this._remove(model, options);
      }
      if (model && ev === 'change:' + model.idAttribute) {
        delete this._byId[model.previous(model.idAttribute)];
        this._byId[model.id] = model;
      }
      this.trigger.apply(this, arguments);
    }

  });

  // Underscore methods that we want to implement on the Collection.
  var methods = ['forEach', 'each', 'map', 'reduce', 'reduceRight', 'find', 'detect',
    'filter', 'select', 'reject', 'every', 'all', 'some', 'any', 'include',
    'contains', 'invoke', 'max', 'min', 'sortBy', 'sortedIndex', 'toArray', 'size',
    'first', 'rest', 'last', 'without', 'indexOf', 'lastIndexOf', 'isEmpty', 'groupBy'];

  // Mix in each Underscore method as a proxy to `Collection#models`.
  _.each(methods, function(method) {
    Backbone.Collection.prototype[method] = function() {
      return _[method].apply(_, [this.models].concat(_.toArray(arguments)));
    };
  });

  // Backbone.Router
  // -------------------

  // Routers map faux-URLs to actions, and fire events when routes are
  // matched. Creating a new one sets its `routes` hash, if not set statically.
  Backbone.Router = function(options) {
    options || (options = {});
    if (options.routes) this.routes = options.routes;
    this._bindRoutes();
    this.initialize.apply(this, arguments);
  };

  // Cached regular expressions for matching named param parts and splatted
  // parts of route strings.
  var namedParam    = /:([\w\d]+)/g;
  var splatParam    = /\*([\w\d]+)/g;
  var escapeRegExp  = /[-[\]{}()+?.,\\^$|#\s]/g;

  // Set up all inheritable **Backbone.Router** properties and methods.
  _.extend(Backbone.Router.prototype, Backbone.Events, {

    // Initialize is an empty function by default. Override it with your own
    // initialization logic.
    initialize : function(){},

    // Manually bind a single named route to a callback. For example:
    //
    //     this.route('search/:query/p:num', 'search', function(query, num) {
    //       ...
    //     });
    //
    route : function(route, name, callback) {
      Backbone.history || (Backbone.history = new Backbone.History);
      if (!_.isRegExp(route)) route = this._routeToRegExp(route);
      Backbone.history.route(route, _.bind(function(fragment) {
        var args = this._extractParameters(route, fragment);
        callback.apply(this, args);
        this.trigger.apply(this, ['route:' + name].concat(args));
      }, this));
    },

    // Simple proxy to `Backbone.history` to save a fragment into the history.
    navigate : function(fragment, triggerRoute) {
      Backbone.history.navigate(fragment, triggerRoute);
    },

    // Bind all defined routes to `Backbone.history`. We have to reverse the
    // order of the routes here to support behavior where the most general
    // routes can be defined at the bottom of the route map.
    _bindRoutes : function() {
      if (!this.routes) return;
      var routes = [];
      for (var route in this.routes) {
        routes.unshift([route, this.routes[route]]);
      }
      for (var i = 0, l = routes.length; i < l; i++) {
        this.route(routes[i][0], routes[i][1], this[routes[i][1]]);
      }
    },

    // Convert a route string into a regular expression, suitable for matching
    // against the current location hash.
    _routeToRegExp : function(route) {
      route = route.replace(escapeRegExp, "\\$&")
                   .replace(namedParam, "([^\/]*)")
                   .replace(splatParam, "(.*?)");
      return new RegExp('^' + route + '$');
    },

    // Given a route, and a URL fragment that it matches, return the array of
    // extracted parameters.
    _extractParameters : function(route, fragment) {
      return route.exec(fragment).slice(1);
    }

  });

  // Backbone.History
  // ----------------

  // Handles cross-browser history management, based on URL fragments. If the
  // browser does not support `onhashchange`, falls back to polling.
  Backbone.History = function() {
    this.handlers = [];
    _.bindAll(this, 'checkUrl');
  };

  // Cached regex for cleaning hashes.
  var hashStrip = /^#*/;

  // Cached regex for detecting MSIE.
  var isExplorer = /msie [\w.]+/;

  // Has the history handling already been started?
  var historyStarted = false;

  // Set up all inheritable **Backbone.History** properties and methods.
  _.extend(Backbone.History.prototype, {

    // The default interval to poll for hash changes, if necessary, is
    // twenty times a second.
    interval: 50,

    // Get the cross-browser normalized URL fragment, either from the URL,
    // the hash, or the override.
    getFragment : function(fragment, forcePushState) {
      if (fragment == null) {
        if (this._hasPushState || forcePushState) {
          fragment = window.location.pathname;
          var search = window.location.search;
          if (search) fragment += search;
          if (fragment.indexOf(this.options.root) == 0) fragment = fragment.substr(this.options.root.length);
        } else {
          fragment = window.location.hash;
        }
      }
      return decodeURIComponent(fragment.replace(hashStrip, ''));
    },

    // Start the hash change handling, returning `true` if the current URL matches
    // an existing route, and `false` otherwise.
    start : function(options) {

      // Figure out the initial configuration. Do we need an iframe?
      // Is pushState desired ... is it available?
      if (historyStarted) throw new Error("Backbone.history has already been started");
      this.options          = _.extend({}, {root: '/'}, this.options, options);
      this._wantsPushState  = !!this.options.pushState;
      this._hasPushState    = !!(this.options.pushState && window.history && window.history.pushState);
      var fragment          = this.getFragment();
      var docMode           = document.documentMode;
      var oldIE             = (isExplorer.exec(navigator.userAgent.toLowerCase()) && (!docMode || docMode <= 7));
      if (oldIE) {
        this.iframe = $('<iframe src="javascript:0" tabindex="-1" />').hide().appendTo('body')[0].contentWindow;
        this.navigate(fragment);
      }

      // Depending on whether we're using pushState or hashes, and whether
      // 'onhashchange' is supported, determine how we check the URL state.
      if (this._hasPushState) {
        $(window).bind('popstate', this.checkUrl);
      } else if ('onhashchange' in window && !oldIE) {
        $(window).bind('hashchange', this.checkUrl);
      } else {
        setInterval(this.checkUrl, this.interval);
      }

      // Determine if we need to change the base url, for a pushState link
      // opened by a non-pushState browser.
      this.fragment = fragment;
      historyStarted = true;
      var loc = window.location;
      var atRoot  = loc.pathname == this.options.root;
      if (this._wantsPushState && !this._hasPushState && !atRoot) {
        this.fragment = this.getFragment(null, true);
        window.location.replace(this.options.root + '#' + this.fragment);
        // Return immediately as browser will do redirect to new url
        return true;
      } else if (this._wantsPushState && this._hasPushState && atRoot && loc.hash) {
        this.fragment = loc.hash.replace(hashStrip, '');
        window.history.replaceState({}, document.title, loc.protocol + '//' + loc.host + this.options.root + this.fragment);
      }

      if (!this.options.silent) {
        return this.loadUrl();
      }
    },

    // Add a route to be tested when the fragment changes. Routes added later may
    // override previous routes.
    route : function(route, callback) {
      this.handlers.unshift({route : route, callback : callback});
    },

    // Checks the current URL to see if it has changed, and if it has,
    // calls `loadUrl`, normalizing across the hidden iframe.
    checkUrl : function(e) {
      var current = this.getFragment();
      if (current == this.fragment && this.iframe) current = this.getFragment(this.iframe.location.hash);
      if (current == this.fragment || current == decodeURIComponent(this.fragment)) return false;
      if (this.iframe) this.navigate(current);
      this.loadUrl() || this.loadUrl(window.location.hash);
    },

    // Attempt to load the current URL fragment. If a route succeeds with a
    // match, returns `true`. If no defined routes matches the fragment,
    // returns `false`.
    loadUrl : function(fragmentOverride) {
      var fragment = this.fragment = this.getFragment(fragmentOverride);
      var matched = _.any(this.handlers, function(handler) {
        if (handler.route.test(fragment)) {
          handler.callback(fragment);
          return true;
        }
      });
      return matched;
    },

    // Save a fragment into the hash history. You are responsible for properly
    // URL-encoding the fragment in advance. This does not trigger
    // a `hashchange` event.
    navigate : function(fragment, triggerRoute) {
      var frag = (fragment || '').replace(hashStrip, '');
      if (this.fragment == frag || this.fragment == decodeURIComponent(frag)) return;
      if (this._hasPushState) {
        var loc = window.location;
        if (frag.indexOf(this.options.root) != 0) frag = this.options.root + frag;
        this.fragment = frag;
        window.history.pushState({}, document.title, loc.protocol + '//' + loc.host + frag);
      } else {
        window.location.hash = this.fragment = frag;
        if (this.iframe && (frag != this.getFragment(this.iframe.location.hash))) {
          this.iframe.document.open().close();
          this.iframe.location.hash = frag;
        }
      }
      if (triggerRoute) this.loadUrl(fragment);
    }

  });

  // Backbone.View
  // -------------

  // Creating a Backbone.View creates its initial element outside of the DOM,
  // if an existing element is not provided...
  Backbone.View = function(options) {
    this.cid = _.uniqueId('view');
    this._configure(options || {});
    this._ensureElement();
    this.delegateEvents();
    this.initialize.apply(this, arguments);
  };

  // Element lookup, scoped to DOM elements within the current view.
  // This should be prefered to global lookups, if you're dealing with
  // a specific view.
  var selectorDelegate = function(selector) {
    return $(selector, this.el);
  };

  // Cached regex to split keys for `delegate`.
  var eventSplitter = /^(\S+)\s*(.*)$/;

  // List of view options to be merged as properties.
  var viewOptions = ['model', 'collection', 'el', 'id', 'attributes', 'className', 'tagName'];

  // Set up all inheritable **Backbone.View** properties and methods.
  _.extend(Backbone.View.prototype, Backbone.Events, {

    // The default `tagName` of a View's element is `"div"`.
    tagName : 'div',

    // Attach the `selectorDelegate` function as the `$` property.
    $       : selectorDelegate,

    // Initialize is an empty function by default. Override it with your own
    // initialization logic.
    initialize : function(){},

    // **render** is the core function that your view should override, in order
    // to populate its element (`this.el`), with the appropriate HTML. The
    // convention is for **render** to always return `this`.
    render : function() {
      return this;
    },

    // Remove this view from the DOM. Note that the view isn't present in the
    // DOM by default, so calling this method may be a no-op.
    remove : function() {
      $(this.el).remove();
      return this;
    },

    // For small amounts of DOM Elements, where a full-blown template isn't
    // needed, use **make** to manufacture elements, one at a time.
    //
    //     var el = this.make('li', {'class': 'row'}, this.model.escape('title'));
    //
    make : function(tagName, attributes, content) {
      var el = document.createElement(tagName);
      if (attributes) $(el).attr(attributes);
      if (content) $(el).html(content);
      return el;
    },

    // Set callbacks, where `this.callbacks` is a hash of
    //
    // *{"event selector": "callback"}*
    //
    //     {
    //       'mousedown .title':  'edit',
    //       'click .button':     'save'
    //     }
    //
    // pairs. Callbacks will be bound to the view, with `this` set properly.
    // Uses event delegation for efficiency.
    // Omitting the selector binds the event to `this.el`.
    // This only works for delegate-able events: not `focus`, `blur`, and
    // not `change`, `submit`, and `reset` in Internet Explorer.
    delegateEvents : function(events) {
      if (!(events || (events = this.events))) return;
      if (_.isFunction(events)) events = events.call(this);
      $(this.el).unbind('.delegateEvents' + this.cid);
      for (var key in events) {
        var method = this[events[key]];
        if (!method) throw new Error('Event "' + events[key] + '" does not exist');
        var match = key.match(eventSplitter);
        var eventName = match[1], selector = match[2];
        method = _.bind(method, this);
        eventName += '.delegateEvents' + this.cid;
        if (selector === '') {
          $(this.el).bind(eventName, method);
        } else {
          $(this.el).delegate(selector, eventName, method);
        }
      }
    },

    // Performs the initial configuration of a View with a set of options.
    // Keys with special meaning *(model, collection, id, className)*, are
    // attached directly to the view.
    _configure : function(options) {
      if (this.options) options = _.extend({}, this.options, options);
      for (var i = 0, l = viewOptions.length; i < l; i++) {
        var attr = viewOptions[i];
        if (options[attr]) this[attr] = options[attr];
      }
      this.options = options;
    },

    // Ensure that the View has a DOM element to render into.
    // If `this.el` is a string, pass it through `$()`, take the first
    // matching element, and re-assign it to `el`. Otherwise, create
    // an element from the `id`, `className` and `tagName` proeprties.
    _ensureElement : function() {
      if (!this.el) {
        var attrs = this.attributes || {};
        if (this.id) attrs.id = this.id;
        if (this.className) attrs['class'] = this.className;
        this.el = this.make(this.tagName, attrs);
      } else if (_.isString(this.el)) {
        this.el = $(this.el).get(0);
      }
    }

  });

  // The self-propagating extend function that Backbone classes use.
  var extend = function (protoProps, classProps) {
    var child = inherits(this, protoProps, classProps);
    child.extend = this.extend;
    return child;
  };

  // Set up inheritance for the model, collection, and view.
  Backbone.Model.extend = Backbone.Collection.extend =
    Backbone.Router.extend = Backbone.View.extend = extend;

  // Map from CRUD to HTTP for our default `Backbone.sync` implementation.
  var methodMap = {
    'create': 'POST',
    'update': 'PUT',
    'delete': 'DELETE',
    'read'  : 'GET'
  };

  // Backbone.sync
  // -------------

  // Override this function to change the manner in which Backbone persists
  // models to the server. You will be passed the type of request, and the
  // model in question. By default, uses makes a RESTful Ajax request
  // to the model's `url()`. Some possible customizations could be:
  //
  // * Use `setTimeout` to batch rapid-fire updates into a single request.
  // * Send up the models as XML instead of JSON.
  // * Persist models via WebSockets instead of Ajax.
  //
  // Turn on `Backbone.emulateHTTP` in order to send `PUT` and `DELETE` requests
  // as `POST`, with a `_method` parameter containing the true HTTP method,
  // as well as all requests with the body as `application/x-www-form-urlencoded` instead of
  // `application/json` with the model in a param named `model`.
  // Useful when interfacing with server-side languages like **PHP** that make
  // it difficult to read the body of `PUT` requests.
  Backbone.sync = function(method, model, options) {
    var type = methodMap[method];

    // Default JSON-request options.
    var params = _.extend({
      type:         type,
      dataType:     'json'
    }, options);

    // Ensure that we have a URL.
    if (!params.url) {
      params.url = getUrl(model) || urlError();
    }

    // Ensure that we have the appropriate request data.
    if (!params.data && model && (method == 'create' || method == 'update')) {
      params.contentType = 'application/json';
      params.data = JSON.stringify(model.toJSON());
    }

    // For older servers, emulate JSON by encoding the request into an HTML-form.
    if (Backbone.emulateJSON) {
      params.contentType = 'application/x-www-form-urlencoded';
      params.data        = params.data ? {model : params.data} : {};
    }

    // For older servers, emulate HTTP by mimicking the HTTP method with `_method`
    // And an `X-HTTP-Method-Override` header.
    if (Backbone.emulateHTTP) {
      if (type === 'PUT' || type === 'DELETE') {
        if (Backbone.emulateJSON) params.data._method = type;
        params.type = 'POST';
        params.beforeSend = function(xhr) {
          xhr.setRequestHeader('X-HTTP-Method-Override', type);
        };
      }
    }

    // Don't process data on a non-GET request.
    if (params.type !== 'GET' && !Backbone.emulateJSON) {
      params.processData = false;
    }

    // Make the request.
    return $.ajax(params);
  };

  // Helpers
  // -------

  // Shared empty constructor function to aid in prototype-chain creation.
  var ctor = function(){};

  // Helper function to correctly set up the prototype chain, for subclasses.
  // Similar to `goog.inherits`, but uses a hash of prototype properties and
  // class properties to be extended.
  var inherits = function(parent, protoProps, staticProps) {
    var child;

    // The constructor function for the new subclass is either defined by you
    // (the "constructor" property in your `extend` definition), or defaulted
    // by us to simply call `super()`.
    if (protoProps && protoProps.hasOwnProperty('constructor')) {
      child = protoProps.constructor;
    } else {
      child = function(){ return parent.apply(this, arguments); };
    }

    // Inherit class (static) properties from parent.
    _.extend(child, parent);

    // Set the prototype chain to inherit from `parent`, without calling
    // `parent`'s constructor function.
    ctor.prototype = parent.prototype;
    child.prototype = new ctor();

    // Add prototype properties (instance properties) to the subclass,
    // if supplied.
    if (protoProps) _.extend(child.prototype, protoProps);

    // Add static properties to the constructor function, if supplied.
    if (staticProps) _.extend(child, staticProps);

    // Correctly set child's `prototype.constructor`.
    child.prototype.constructor = child;

    // Set a convenience property in case the parent's prototype is needed later.
    child.__super__ = parent.prototype;

    return child;
  };

  // Helper function to get a URL from a Model or Collection as a property
  // or as a function.
  var getUrl = function(object) {
    if (!(object && object.url)) return null;
    return _.isFunction(object.url) ? object.url() : object.url;
  };

  // Throw an error when a URL is needed, and none is supplied.
  var urlError = function() {
    throw new Error('A "url" property or function must be specified');
  };

  // Wrap an optional error callback with a fallback error event.
  var wrapError = function(onError, model, options) {
    return function(resp) {
      if (onError) {
        onError(model, resp, options);
      } else {
        model.trigger('error', model, resp, options);
      }
    };
  };

  // Helper function to escape a string for HTML rendering.
  var escapeHTML = function(string) {
    return string.replace(/&(?!\w+;|#\d+;|#x[\da-f]+;)/gi, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#x27;').replace(/\//g,'&#x2F;');
  };

}).call(this);
/**
 * Castor - a cross site POSTing JavaScript logging library for Loggly
 *
 * Copyright (c) 2011 Loggly, Inc.
 * All rights reserved.
 *
 * Author: Kord Campbell <kord@loggly.com>
 * Date: May 2, 2011
 *
 * Uses methods from janky.post, copyright(c) 2011 Thomas Rampelberg <thomas@saunter.org>
 *
 * Sample usage (replace with your own Loggly HTTP input URL):

  <script src="/js/loggly.js" type="text/javascript"></script>
  <script type="text/javascript">
    window.onload=function(){
      castor = new loggly({ url: 'http://logs.loggly.com/inputs/a4e839e9-4227-49aa-9d28-e18e5ba5a818?rt=1', level: 'WARN'});
      castor.log("url="+window.location.href + " browser=" + castor.user_agent + " height=" + castor.browser_size.height);
    }
  </script>

 */

(function() {
  this.loggly = function(opts) {
    this.user_agent = get_agent();
    this.browser_size = get_size();
    log_methods = {'error': 5, 'warn': 4, 'info': 3, 'debug': 2, 'log': 1};
    if (!opts.url) throw new Error("Please include a Loggly HTTP URL.");
    if (!opts.level) {
      this.level = log_methods['info'];
    } else {
      this.level = log_methods[opts.level];
    }
    this.log = function(data) {
      if (log_methods['log'] == this.level) {
        opts.data = data;
        janky(opts);
      }
    };
    this.debug = function(data) {
      if (log_methods['debug'] >= this.level) {
        opts.data = data;
        janky(opts);
      }
    };
    this.info = function(data) {
      if (log_methods['info'] >= this.level) {
        opts.data = data;
        janky(opts);
      }
    };
    this.warn = function(data) {
      if (log_methods['warn'] >= this.level) {
        opts.data = data;
        janky(opts);
      }
    };
    this.error = function(data) {
      if (log_methods['error'] >= this.level) {
        opts.data = data;
        janky(opts);
      }
    };
  };
  this.janky = function(opts) {
    janky._form(function(iframe, form) {
      form.setAttribute("action", opts.url);
      form.setAttribute("method", "post");
      janky._input(iframe, form, opts.data);
      form.submit();
	  setTimeout(function(){
        document.body.removeChild(iframe);
      }, 2000);
    });
  };
  this.janky._form = function(cb) {
    var iframe = document.createElement("iframe");
    document.body.appendChild(iframe);
    iframe.style.display = "none";
    setTimeout(function() {
      var form = iframe.contentWindow.document.createElement("form");
      iframe.contentWindow.document.body.appendChild(form);
      cb(iframe, form);
    }, 0);
  };
  this.janky._input = function(iframe, form, data) {
    var inp = iframe.contentWindow.document.createElement("input");
    inp.setAttribute("type", "hidden");
    inp.setAttribute("name", "source");
    inp.value = "castor " + data;
    form.appendChild(inp);
  };
  this.get_agent = function () {
    return navigator.appCodeName + navigator.appName + navigator.appVersion;
  };
  this.get_size = function () {
    var width = 0; var height = 0;
    if( typeof( window.innerWidth ) == 'number' ) {
      width = window.innerWidth; height = window.innerHeight;
    } else if( document.documentElement && ( document.documentElement.clientWidth || document.documentElement.clientHeight ) ) {
      width = document.documentElement.clientWidth; height = document.documentElement.clientHeight;
    } else if( document.body && ( document.body.clientWidth || document.body.clientHeight ) ) {
      width = document.body.clientWidth; height = document.body.clientHeight;
    }
    return {'height': height, 'width': width};
  };
})();


jsworld={};jsworld.formatIsoDateTime=function(a,b){if(typeof a==="undefined")a=new Date;if(typeof b==="undefined")b=false;var c=jsworld.formatIsoDate(a)+" "+jsworld.formatIsoTime(a);if(b){var d=a.getHours()-a.getUTCHours();var e=Math.abs(d);var f=a.getUTCMinutes();var g=a.getMinutes();if(g!=f&&f<30&&d<0)e--;if(g!=f&&f>30&&d>0)e--;var h;if(g!=f)h=":30";else h=":00";var i;if(e<10)i="0"+e+h;else i=""+e+h;if(d<0)i="-"+i;else i="+"+i;c=c+i}return c};jsworld.formatIsoDate=function(a){if(typeof a==="undefined")a=new Date;var b=a.getFullYear();var c=a.getMonth()+1;var d=a.getDate();return b+"-"+jsworld._zeroPad(c,2)+"-"+jsworld._zeroPad(d,2)};jsworld.formatIsoTime=function(a){if(typeof a==="undefined")a=new Date;var b=a.getHours();var c=a.getMinutes();var d=a.getSeconds();return jsworld._zeroPad(b,2)+":"+jsworld._zeroPad(c,2)+":"+jsworld._zeroPad(d,2)};jsworld.parseIsoDateTime=function(a){if(typeof a!="string")throw"Error: The parameter must be a string";var b=a.match(/^(\d\d\d\d)-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)/);if(b===null)b=a.match(/^(\d\d\d\d)(\d\d)(\d\d)[T ](\d\d)(\d\d)(\d\d)/);if(b===null)b=a.match(/^(\d\d\d\d)-(\d\d)-(\d\d)[T ](\d\d)(\d\d)(\d\d)/);if(b===null)b=a.match(/^(\d\d\d\d)-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)/);if(b===null)throw"Error: Invalid ISO-8601 date/time string";var c=parseInt(b[1],10);var d=parseInt(b[2],10);var e=parseInt(b[3],10);var f=parseInt(b[4],10);var g=parseInt(b[5],10);var h=parseInt(b[6],10);if(d<1||d>12||e<1||e>31||f<0||f>23||g<0||g>59||h<0||h>59)throw"Error: Invalid ISO-8601 date/time value";var i=new Date(c,d-1,e,f,g,h);if(i.getDate()!=e||i.getMonth()+1!=d)throw"Error: Invalid date";return i};jsworld.parseIsoDate=function(a){if(typeof a!="string")throw"Error: The parameter must be a string";var b=a.match(/^(\d\d\d\d)-(\d\d)-(\d\d)/);if(b===null)b=a.match(/^(\d\d\d\d)(\d\d)(\d\d)/);if(b===null)throw"Error: Invalid ISO-8601 date string";var c=parseInt(b[1],10);var d=parseInt(b[2],10);var e=parseInt(b[3],10);if(d<1||d>12||e<1||e>31)throw"Error: Invalid ISO-8601 date value";var f=new Date(c,d-1,e);if(f.getDate()!=e||f.getMonth()+1!=d)throw"Error: Invalid date";return f};jsworld.parseIsoTime=function(a){if(typeof a!="string")throw"Error: The parameter must be a string";var b=a.match(/^(\d\d):(\d\d):(\d\d)/);if(b===null)b=a.match(/^(\d\d)(\d\d)(\d\d)/);if(b===null)throw"Error: Invalid ISO-8601 date/time string";var c=parseInt(b[1],10);var d=parseInt(b[2],10);var e=parseInt(b[3],10);if(c<0||c>23||d<0||d>59||e<0||e>59)throw"Error: Invalid ISO-8601 time value";return new Date(0,0,0,c,d,e)};jsworld._trim=function(a){var b=" \n\r\t\f \u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u200b\u2028\u2029\u3000";for(var c=0;c<a.length;c++){if(b.indexOf(a.charAt(c))===-1){a=a.substring(c);break}}for(c=a.length-1;c>=0;c--){if(b.indexOf(a.charAt(c))===-1){a=a.substring(0,c+1);break}}return b.indexOf(a.charAt(0))===-1?a:""};jsworld._isNumber=function(a){if(typeof a=="number")return true;if(typeof a!="string")return false;var b=a+"";return/^-?(\d+|\d*\.\d+)$/.test(b)};jsworld._isInteger=function(a){if(typeof a!="number"&&typeof a!="string")return false;var b=a+"";return/^-?\d+$/.test(b)};jsworld._isFloat=function(a){if(typeof a!="number"&&typeof a!="string")return false;var b=a+"";return/^-?\.\d+?$/.test(b)};jsworld._hasOption=function(a,b){if(typeof a!="string"||typeof b!="string")return false;if(b.indexOf(a)!=-1)return true;else return false};jsworld._stringReplaceAll=function(a,b,c){var d;if(b.length==1&&c.length==1){d="";for(var e=0;e<a.length;e++){if(a.charAt(e)==b.charAt(0))d=d+c.charAt(0);else d=d+a.charAt(e)}return d}else{d=a;var f=d.indexOf(b);while(f!=-1){d=d.replace(b,c);f=d.indexOf(b)}return d}};jsworld._stringStartsWith=function(a,b){if(a.length<b.length)return false;for(var c=0;c<b.length;c++){if(a.charAt(c)!=b.charAt(c))return false}return true};jsworld._getPrecision=function(a){if(typeof a!="string")return-1;var b=a.match(/\.(\d)/);if(b)return parseInt(b[1],10);else return-1};jsworld._splitNumber=function(a){if(typeof a=="number")a=a+"";var b={};if(a.charAt(0)=="-")a=a.substring(1);var c=a.split(".");if(!c[1])c[1]="";b.integer=c[0];b.fraction=c[1];return b};jsworld._formatIntegerPart=function(a,b,c){if(c==""||b=="-1")return a;var d=b.split(";");var e="";var f=a.length;var g;while(f>0){if(d.length>0)g=parseInt(d.shift(),10);if(isNaN(g))throw"Error: Invalid grouping";if(g==-1){e=a.substring(0,f)+e;break}f-=g;if(f<1){e=a.substring(0,f+g)+e;break}e=c+a.substring(f,f+g)+e}return e};jsworld._formatFractionPart=function(a,b){for(var c=0;a.length<b;c++)a=a+"0";return a};jsworld._zeroPad=function(a,b){var c=a+"";while(c.length<b)c="0"+c;return c};jsworld._spacePad=function(a,b){var c=a+"";while(c.length<b)c=" "+c;return c};jsworld.Locale=function(a){this._className="jsworld.Locale";this._parseList=function(a,b){var c=[];if(a==null){throw"Names not defined"}else if(typeof a=="object"){c=a}else if(typeof a=="string"){c=a.split(";",b);for(var d=0;d<c.length;d++){if(c[d][0]=='"'&&c[d][c[d].length-1]=='"')c[d]=c[d].slice(1,-1);else throw"Missing double quotes"}}else{throw"Names must be an array or a string"}if(c.length!=b)throw"Expected "+b+" items, got "+c.length;return c};this._validateFormatString=function(a){if(typeof a=="string"&&a.length>0)return a;else throw"Empty or no string"};if(a==null||typeof a!="object")throw"Error: Invalid/missing locale properties";if(typeof a.decimal_point!="string")throw"Error: Invalid/missing decimal_point property";this.decimal_point=a.decimal_point;if(typeof a.thousands_sep!="string")throw"Error: Invalid/missing thousands_sep property";this.thousands_sep=a.thousands_sep;if(typeof a.grouping!="string")throw"Error: Invalid/missing grouping property";this.grouping=a.grouping;if(typeof a.int_curr_symbol!="string")throw"Error: Invalid/missing int_curr_symbol property";if(!/[A-Za-z]{3}.?/.test(a.int_curr_symbol))throw"Error: Invalid int_curr_symbol property";this.int_curr_symbol=a.int_curr_symbol;if(typeof a.currency_symbol!="string")throw"Error: Invalid/missing currency_symbol property";this.currency_symbol=a.currency_symbol;if(typeof a.frac_digits!="number"&&a.frac_digits<0)throw"Error: Invalid/missing frac_digits property";this.frac_digits=a.frac_digits;if(a.mon_decimal_point===null||a.mon_decimal_point==""){if(this.frac_digits>0)throw"Error: Undefined mon_decimal_point property";else a.mon_decimal_point=""}if(typeof a.mon_decimal_point!="string")throw"Error: Invalid/missing mon_decimal_point property";this.mon_decimal_point=a.mon_decimal_point;if(typeof a.mon_thousands_sep!="string")throw"Error: Invalid/missing mon_thousands_sep property";this.mon_thousands_sep=a.mon_thousands_sep;if(typeof a.mon_grouping!="string")throw"Error: Invalid/missing mon_grouping property";this.mon_grouping=a.mon_grouping;if(typeof a.positive_sign!="string")throw"Error: Invalid/missing positive_sign property";this.positive_sign=a.positive_sign;if(typeof a.negative_sign!="string")throw"Error: Invalid/missing negative_sign property";this.negative_sign=a.negative_sign;if(a.p_cs_precedes!==0&&a.p_cs_precedes!==1)throw"Error: Invalid/missing p_cs_precedes property, must be 0 or 1";this.p_cs_precedes=a.p_cs_precedes;if(a.n_cs_precedes!==0&&a.n_cs_precedes!==1)throw"Error: Invalid/missing n_cs_precedes, must be 0 or 1";this.n_cs_precedes=a.n_cs_precedes;if(a.p_sep_by_space!==0&&a.p_sep_by_space!==1&&a.p_sep_by_space!==2)throw"Error: Invalid/missing p_sep_by_space property, must be 0, 1 or 2";this.p_sep_by_space=a.p_sep_by_space;if(a.n_sep_by_space!==0&&a.n_sep_by_space!==1&&a.n_sep_by_space!==2)throw"Error: Invalid/missing n_sep_by_space property, must be 0, 1, or 2";this.n_sep_by_space=a.n_sep_by_space;if(a.p_sign_posn!==0&&a.p_sign_posn!==1&&a.p_sign_posn!==2&&a.p_sign_posn!==3&&a.p_sign_posn!==4)throw"Error: Invalid/missing p_sign_posn property, must be 0, 1, 2, 3 or 4";this.p_sign_posn=a.p_sign_posn;if(a.n_sign_posn!==0&&a.n_sign_posn!==1&&a.n_sign_posn!==2&&a.n_sign_posn!==3&&a.n_sign_posn!==4)throw"Error: Invalid/missing n_sign_posn property, must be 0, 1, 2, 3 or 4";this.n_sign_posn=a.n_sign_posn;if(typeof a.int_frac_digits!="number"&&a.int_frac_digits<0)throw"Error: Invalid/missing int_frac_digits property";this.int_frac_digits=a.int_frac_digits;if(a.int_p_cs_precedes!==0&&a.int_p_cs_precedes!==1)throw"Error: Invalid/missing int_p_cs_precedes property, must be 0 or 1";this.int_p_cs_precedes=a.int_p_cs_precedes;if(a.int_n_cs_precedes!==0&&a.int_n_cs_precedes!==1)throw"Error: Invalid/missing int_n_cs_precedes property, must be 0 or 1";this.int_n_cs_precedes=a.int_n_cs_precedes;if(a.int_p_sep_by_space!==0&&a.int_p_sep_by_space!==1&&a.int_p_sep_by_space!==2)throw"Error: Invalid/missing int_p_sep_by_spacev, must be 0, 1 or 2";this.int_p_sep_by_space=a.int_p_sep_by_space;if(a.int_n_sep_by_space!==0&&a.int_n_sep_by_space!==1&&a.int_n_sep_by_space!==2)throw"Error: Invalid/missing int_n_sep_by_space property, must be 0, 1, or 2";this.int_n_sep_by_space=a.int_n_sep_by_space;if(a.int_p_sign_posn!==0&&a.int_p_sign_posn!==1&&a.int_p_sign_posn!==2&&a.int_p_sign_posn!==3&&a.int_p_sign_posn!==4)throw"Error: Invalid/missing int_p_sign_posn property, must be 0, 1, 2, 3 or 4";this.int_p_sign_posn=a.int_p_sign_posn;if(a.int_n_sign_posn!==0&&a.int_n_sign_posn!==1&&a.int_n_sign_posn!==2&&a.int_n_sign_posn!==3&&a.int_n_sign_posn!==4)throw"Error: Invalid/missing int_n_sign_posn property, must be 0, 1, 2, 3 or 4";this.int_n_sign_posn=a.int_n_sign_posn;if(a==null||typeof a!="object")throw"Error: Invalid/missing time locale properties";try{this.abday=this._parseList(a.abday,7)}catch(b){throw"Error: Invalid abday property: "+b}try{this.day=this._parseList(a.day,7)}catch(b){throw"Error: Invalid day property: "+b}try{this.abmon=this._parseList(a.abmon,12)}catch(b){throw"Error: Invalid abmon property: "+b}try{this.mon=this._parseList(a.mon,12)}catch(b){throw"Error: Invalid mon property: "+b}try{this.d_fmt=this._validateFormatString(a.d_fmt)}catch(b){throw"Error: Invalid d_fmt property: "+b}try{this.t_fmt=this._validateFormatString(a.t_fmt)}catch(b){throw"Error: Invalid t_fmt property: "+b}try{this.d_t_fmt=this._validateFormatString(a.d_t_fmt)}catch(b){throw"Error: Invalid d_t_fmt property: "+b}try{var c=this._parseList(a.am_pm,2);this.am=c[0];this.pm=c[1]}catch(b){this.am="";this.pm=""}this.getAbbreviatedWeekdayName=function(a){if(typeof a=="undefined"||a===null)return this.abday;if(!jsworld._isInteger(a)||a<0||a>6)throw"Error: Invalid weekday argument, must be an integer [0..6]";return this.abday[a]};this.getWeekdayName=function(a){if(typeof a=="undefined"||a===null)return this.day;if(!jsworld._isInteger(a)||a<0||a>6)throw"Error: Invalid weekday argument, must be an integer [0..6]";return this.day[a]};this.getAbbreviatedMonthName=function(a){if(typeof a=="undefined"||a===null)return this.abmon;if(!jsworld._isInteger(a)||a<0||a>11)throw"Error: Invalid month argument, must be an integer [0..11]";return this.abmon[a]};this.getMonthName=function(a){if(typeof a=="undefined"||a===null)return this.mon;if(!jsworld._isInteger(a)||a<0||a>11)throw"Error: Invalid month argument, must be an integer [0..11]";return this.mon[a]};this.getDecimalPoint=function(){return this.decimal_point};this.getCurrencySymbol=function(){return this.currency_symbol};this.getIntCurrencySymbol=function(){return this.int_curr_symbol.substring(0,3)};this.currencySymbolPrecedes=function(){if(this.p_cs_precedes==1)return true;else return false};this.intCurrencySymbolPrecedes=function(){if(this.int_p_cs_precedes==1)return true;else return false};this.getMonetaryDecimalPoint=function(){return this.mon_decimal_point};this.getFractionalDigits=function(){return this.frac_digits};this.getIntFractionalDigits=function(){return this.int_frac_digits}};jsworld.NumericFormatter=function(a){if(typeof a!="object"||a._className!="jsworld.Locale")throw"Constructor error: You must provide a valid jsworld.Locale instance";this.lc=a;this.format=function(a,b){if(typeof a=="string")a=jsworld._trim(a);if(!jsworld._isNumber(a))throw"Error: The input is not a number";var c=parseFloat(a,10);var d=jsworld._getPrecision(b);if(d!=-1)c=Math.round(c*Math.pow(10,d))/Math.pow(10,d);var e=jsworld._splitNumber(String(c));var f;if(c===0)f="0";else f=jsworld._hasOption("^",b)?e.integer:jsworld._formatIntegerPart(e.integer,this.lc.grouping,this.lc.thousands_sep);var g=d!=-1?jsworld._formatFractionPart(e.fraction,d):e.fraction;var h=g.length?f+this.lc.decimal_point+g:f;if(jsworld._hasOption("~",b)||c===0){return h}else{if(jsworld._hasOption("+",b)||c<0){if(c>0)return"+"+h;else if(c<0)return"-"+h;else return h}else{return h}}}};jsworld.DateTimeFormatter=function(a){if(typeof a!="object"||a._className!="jsworld.Locale")throw"Constructor error: You must provide a valid jsworld.Locale instance.";this.lc=a;this.formatDate=function(a){var b=null;if(typeof a=="string"){try{b=jsworld.parseIsoDate(a)}catch(c){b=jsworld.parseIsoDateTime(a)}}else if(a!==null&&typeof a=="object"){b=a}else{throw"Error: Invalid date argument, must be a Date object or an ISO-8601 date/time string"}return this._applyFormatting(b,this.lc.d_fmt)};this.formatTime=function(a){var b=null;if(typeof a=="string"){try{b=jsworld.parseIsoTime(a)}catch(c){b=jsworld.parseIsoDateTime(a)}}else if(a!==null&&typeof a=="object"){b=a}else{throw"Error: Invalid date argument, must be a Date object or an ISO-8601 date/time string"}return this._applyFormatting(b,this.lc.t_fmt)};this.formatDateTime=function(a){var b=null;if(typeof a=="string"){b=jsworld.parseIsoDateTime(a)}else if(a!==null&&typeof a=="object"){b=a}else{throw"Error: Invalid date argument, must be a Date object or an ISO-8601 date/time string"}return this._applyFormatting(b,this.lc.d_t_fmt)};this._applyFormatting=function(a,b){b=b.replace(/%%/g,"%");b=b.replace(/%a/g,this.lc.abday[a.getDay()]);b=b.replace(/%A/g,this.lc.day[a.getDay()]);b=b.replace(/%b/g,this.lc.abmon[a.getMonth()]);b=b.replace(/%B/g,this.lc.mon[a.getMonth()]);b=b.replace(/%d/g,jsworld._zeroPad(a.getDate(),2));b=b.replace(/%e/g,jsworld._spacePad(a.getDate(),2));b=b.replace(/%F/g,a.getFullYear()+"-"+jsworld._zeroPad(a.getMonth()+1,2)+"-"+jsworld._zeroPad(a.getDate(),2));b=b.replace(/%h/g,this.lc.abmon[a.getMonth()]);b=b.replace(/%H/g,jsworld._zeroPad(a.getHours(),2));b=b.replace(/%I/g,jsworld._zeroPad(this._hours12(a.getHours()),2));b=b.replace(/%k/g,a.getHours());b=b.replace(/%l/g,this._hours12(a.getHours()));b=b.replace(/%m/g,jsworld._zeroPad(a.getMonth()+1,2));b=b.replace(/%n/g,"\n");b=b.replace(/%M/g,jsworld._zeroPad(a.getMinutes(),2));b=b.replace(/%p/g,this._getAmPm(a.getHours()));b=b.replace(/%P/g,this._getAmPm(a.getHours()).toLocaleLowerCase());b=b.replace(/%R/g,jsworld._zeroPad(a.getHours(),2)+":"+jsworld._zeroPad(a.getMinutes(),2));b=b.replace(/%S/g,jsworld._zeroPad(a.getSeconds(),2));b=b.replace(/%T/g,jsworld._zeroPad(a.getHours(),2)+":"+jsworld._zeroPad(a.getMinutes(),2)+":"+jsworld._zeroPad(a.getSeconds(),2));b=b.replace(/%w/g,this.lc.day[a.getDay()]);b=b.replace(/%y/g,(new String(a.getFullYear())).substring(2));b=b.replace(/%Y/g,a.getFullYear());b=b.replace(/%Z/g,"");b=b.replace(/%[a-zA-Z]/g,"");return b};this._hours12=function(a){if(a===0)return 12;else if(a>12)return a-12;else return a};this._getAmPm=function(a){if(a===0||a>12)return this.lc.pm;else return this.lc.am}};jsworld.MonetaryFormatter=function(a,b,c){if(typeof a!="object"||a._className!="jsworld.Locale")throw"Constructor error: You must provide a valid jsworld.Locale instance";this.lc=a;this.currencyFractionDigits={AFN:0,ALL:0,AMD:0,BHD:3,BIF:0,BYR:0,CLF:0,CLP:0,COP:0,CRC:0,DJF:0,GNF:0,GYD:0,HUF:0,IDR:0,IQD:0,IRR:0,ISK:0,JOD:3,JPY:0,KMF:0,KRW:0,KWD:3,LAK:0,LBP:0,LYD:3,MGA:0,MMK:0,MNT:0,MRO:0,MUR:0,OMR:3,PKR:0,PYG:0,RSD:0,RWF:0,SLL:0,SOS:0,STD:0,SYP:0,TND:3,TWD:0,TZS:0,UGX:0,UZS:0,VND:0,VUV:0,XAF:0,XOF:0,XPF:0,YER:0,ZMK:0};if(typeof b=="string"){this.currencyCode=b.toUpperCase();var d=this.currencyFractionDigits[this.currencyCode];if(typeof d!="number")d=2;this.lc.frac_digits=d;this.lc.int_frac_digits=d}else{this.currencyCode=this.lc.int_curr_symbol.substring(0,3).toUpperCase()}this.intSep=this.lc.int_curr_symbol.charAt(3);if(this.currencyCode==this.lc.int_curr_symbol.substring(0,3)){this.internationalFormatting=false;this.curSym=this.lc.currency_symbol}else{if(typeof c=="string"){this.curSym=c;this.internationalFormatting=false}else{this.internationalFormatting=true}}this.getCurrencySymbol=function(){return this.curSym};this.currencySymbolPrecedes=function(a){if(typeof a=="string"&&a=="i"){if(this.lc.int_p_cs_precedes==1)return true;else return false}else{if(this.internationalFormatting){if(this.lc.int_p_cs_precedes==1)return true;else return false}else{if(this.lc.p_cs_precedes==1)return true;else return false}}};this.getDecimalPoint=function(){return this.lc.mon_decimal_point};this.getFractionalDigits=function(a){if(typeof a=="string"&&a=="i"){return this.lc.int_frac_digits}else{if(this.internationalFormatting)return this.lc.int_frac_digits;else return this.lc.frac_digits}};this.format=function(a,b){var c;if(typeof a=="string"){a=jsworld._trim(a);c=parseFloat(a);if(typeof c!="number"||isNaN(c))throw"Error: Amount string not a number"}else if(typeof a=="number"){c=a}else{throw"Error: Amount not a number"}var d=jsworld._getPrecision(b);if(d==-1){if(this.internationalFormatting||jsworld._hasOption("i",b))d=this.lc.int_frac_digits;else d=this.lc.frac_digits}c=Math.round(c*Math.pow(10,d))/Math.pow(10,d);var e=jsworld._splitNumber(String(c));var f;if(c===0)f="0";else f=jsworld._hasOption("^",b)?e.integer:jsworld._formatIntegerPart(e.integer,this.lc.mon_grouping,this.lc.mon_thousands_sep);var g;if(d==-1){if(this.internationalFormatting||jsworld._hasOption("i",b))g=jsworld._formatFractionPart(e.fraction,this.lc.int_frac_digits);else g=jsworld._formatFractionPart(e.fraction,this.lc.frac_digits)}else{g=jsworld._formatFractionPart(e.fraction,d)}var h;if(this.lc.frac_digits>0||g.length)h=f+this.lc.mon_decimal_point+g;else h=f;if(jsworld._hasOption("~",b)){return h}else{var i=jsworld._hasOption("!",b)?true:false;var j=c<0?"-":"+";if(this.internationalFormatting||jsworld._hasOption("i",b)){if(i)return this._formatAsInternationalCurrencyWithNoSym(j,h);else return this._formatAsInternationalCurrency(j,h)}else{if(i)return this._formatAsLocalCurrencyWithNoSym(j,h);else return this._formatAsLocalCurrency(j,h)}}};this._formatAsLocalCurrency=function(a,b){if(a=="+"){if(this.lc.p_sign_posn===0&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===0){return"("+b+this.curSym+")"}else if(this.lc.p_sign_posn===0&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===1){return"("+this.curSym+b+")"}else if(this.lc.p_sign_posn===0&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===0){return"("+b+" "+this.curSym+")"}else if(this.lc.p_sign_posn===0&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===1){return"("+this.curSym+" "+b+")"}else if(this.lc.p_sign_posn===1&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===0){return this.lc.positive_sign+b+this.curSym}else if(this.lc.p_sign_posn===1&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+this.curSym+b}else if(this.lc.p_sign_posn===1&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===0){return this.lc.positive_sign+b+" "+this.curSym}else if(this.lc.p_sign_posn===1&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+this.curSym+" "+b}else if(this.lc.p_sign_posn===1&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===0){return this.lc.positive_sign+" "+b+this.curSym}else if(this.lc.p_sign_posn===1&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+" "+this.curSym+b}else if(this.lc.p_sign_posn===2&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===0){return b+this.curSym+this.lc.positive_sign}else if(this.lc.p_sign_posn===2&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===1){return this.curSym+b+this.lc.positive_sign}else if(this.lc.p_sign_posn===2&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===0){return b+" "+this.curSym+this.lc.positive_sign}else if(this.lc.p_sign_posn===2&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===1){return this.curSym+" "+b+this.lc.positive_sign}else if(this.lc.p_sign_posn===2&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===0){return b+this.curSym+" "+this.lc.positive_sign}else if(this.lc.p_sign_posn===2&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===1){return this.curSym+b+" "+this.lc.positive_sign}else if(this.lc.p_sign_posn===3&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===0){return b+this.lc.positive_sign+this.curSym}else if(this.lc.p_sign_posn===3&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+this.curSym+b}else if(this.lc.p_sign_posn===3&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===0){return b+" "+this.lc.positive_sign+this.curSym}else if(this.lc.p_sign_posn===3&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+this.curSym+" "+b}else if(this.lc.p_sign_posn===3&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===0){return b+this.lc.positive_sign+" "+this.curSym}else if(this.lc.p_sign_posn===3&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+" "+this.curSym+b}else if(this.lc.p_sign_posn===4&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===0){return b+this.curSym+this.lc.positive_sign}else if(this.lc.p_sign_posn===4&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===1){return this.curSym+this.lc.positive_sign+b}else if(this.lc.p_sign_posn===4&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===0){return b+" "+this.curSym+this.lc.positive_sign}else if(this.lc.p_sign_posn===4&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===1){return this.curSym+this.lc.positive_sign+" "+b}else if(this.lc.p_sign_posn===4&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===0){return b+this.curSym+" "+this.lc.positive_sign}else if(this.lc.p_sign_posn===4&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===1){return this.curSym+" "+this.lc.positive_sign+b}}else if(a=="-"){if(this.lc.n_sign_posn===0&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===0){return"("+b+this.curSym+")"}else if(this.lc.n_sign_posn===0&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===1){return"("+this.curSym+b+")"}else if(this.lc.n_sign_posn===0&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===0){return"("+b+" "+this.curSym+")"}else if(this.lc.n_sign_posn===0&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===1){return"("+this.curSym+" "+b+")"}else if(this.lc.n_sign_posn===1&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===0){return this.lc.negative_sign+b+this.curSym}else if(this.lc.n_sign_posn===1&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+this.curSym+b}else if(this.lc.n_sign_posn===1&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===0){return this.lc.negative_sign+b+" "+this.curSym}else if(this.lc.n_sign_posn===1&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+this.curSym+" "+b}else if(this.lc.n_sign_posn===1&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===0){return this.lc.negative_sign+" "+b+this.curSym}else if(this.lc.n_sign_posn===1&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+" "+this.curSym+b}else if(this.lc.n_sign_posn===2&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===0){return b+this.curSym+this.lc.negative_sign}else if(this.lc.n_sign_posn===2&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===1){return this.curSym+b+this.lc.negative_sign}else if(this.lc.n_sign_posn===2&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===0){return b+" "+this.curSym+this.lc.negative_sign}else if(this.lc.n_sign_posn===2&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===1){return this.curSym+" "+b+this.lc.negative_sign}else if(this.lc.n_sign_posn===2&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===0){return b+this.curSym+" "+this.lc.negative_sign}else if(this.lc.n_sign_posn===2&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===1){return this.curSym+b+" "+this.lc.negative_sign}else if(this.lc.n_sign_posn===3&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===0){return b+this.lc.negative_sign+this.curSym}else if(this.lc.n_sign_posn===3&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+this.curSym+b}else if(this.lc.n_sign_posn===3&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===0){return b+" "+this.lc.negative_sign+this.curSym}else if(this.lc.n_sign_posn===3&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+this.curSym+" "+b}else if(this.lc.n_sign_posn===3&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===0){return b+this.lc.negative_sign+" "+this.curSym}else if(this.lc.n_sign_posn===3&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+" "+this.curSym+b}else if(this.lc.n_sign_posn===4&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===0){return b+this.curSym+this.lc.negative_sign}else if(this.lc.n_sign_posn===4&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===1){return this.curSym+this.lc.negative_sign+b}else if(this.lc.n_sign_posn===4&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===0){return b+" "+this.curSym+this.lc.negative_sign}else if(this.lc.n_sign_posn===4&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===1){return this.curSym+this.lc.negative_sign+" "+b}else if(this.lc.n_sign_posn===4&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===0){return b+this.curSym+" "+this.lc.negative_sign}else if(this.lc.n_sign_posn===4&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===1){return this.curSym+" "+this.lc.negative_sign+b}}throw"Error: Invalid POSIX LC MONETARY definition"};this._formatAsInternationalCurrency=function(a,b){if(a=="+"){if(this.lc.int_p_sign_posn===0&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===0){return"("+b+this.currencyCode+")"}else if(this.lc.int_p_sign_posn===0&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===1){return"("+this.currencyCode+b+")"}else if(this.lc.int_p_sign_posn===0&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===0){return"("+b+this.intSep+this.currencyCode+")"}else if(this.lc.int_p_sign_posn===0&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===1){return"("+this.currencyCode+this.intSep+b+")"}else if(this.lc.int_p_sign_posn===1&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===0){return this.lc.positive_sign+b+this.currencyCode}else if(this.lc.int_p_sign_posn===1&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+this.currencyCode+b}else if(this.lc.int_p_sign_posn===1&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===0){return this.lc.positive_sign+b+this.intSep+this.currencyCode}else if(this.lc.int_p_sign_posn===1&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+this.currencyCode+this.intSep+b}else if(this.lc.int_p_sign_posn===1&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===0){return this.lc.positive_sign+this.intSep+b+this.currencyCode}else if(this.lc.int_p_sign_posn===1&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+this.intSep+this.currencyCode+b}else if(this.lc.int_p_sign_posn===2&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===0){return b+this.currencyCode+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===2&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===1){return this.currencyCode+b+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===2&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===0){return b+this.intSep+this.currencyCode+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===2&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===1){return this.currencyCode+this.intSep+b+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===2&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===0){return b+this.currencyCode+this.intSep+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===2&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===1){return this.currencyCode+b+this.intSep+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===3&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===0){return b+this.lc.positive_sign+this.currencyCode}else if(this.lc.int_p_sign_posn===3&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+this.currencyCode+b}else if(this.lc.int_p_sign_posn===3&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===0){return b+this.intSep+this.lc.positive_sign+this.currencyCode}else if(this.lc.int_p_sign_posn===3&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+this.currencyCode+this.intSep+b}else if(this.lc.int_p_sign_posn===3&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===0){return b+this.lc.positive_sign+this.intSep+this.currencyCode}else if(this.lc.int_p_sign_posn===3&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+this.intSep+this.currencyCode+b}else if(this.lc.int_p_sign_posn===4&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===0){return b+this.currencyCode+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===4&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===1){return this.currencyCode+this.lc.positive_sign+b}else if(this.lc.int_p_sign_posn===4&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===0){return b+this.intSep+this.currencyCode+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===4&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===1){return this.currencyCode+this.lc.positive_sign+this.intSep+b}else if(this.lc.int_p_sign_posn===4&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===0){return b+this.currencyCode+this.intSep+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===4&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===1){return this.currencyCode+this.intSep+this.lc.positive_sign+b}}else if(a=="-"){if(this.lc.int_n_sign_posn===0&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===0){return"("+b+this.currencyCode+")"}else if(this.lc.int_n_sign_posn===0&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===1){return"("+this.currencyCode+b+")"}else if(this.lc.int_n_sign_posn===0&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===0){return"("+b+this.intSep+this.currencyCode+")"}else if(this.lc.int_n_sign_posn===0&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===1){return"("+this.currencyCode+this.intSep+b+")"}else if(this.lc.int_n_sign_posn===1&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===0){return this.lc.negative_sign+b+this.currencyCode}else if(this.lc.int_n_sign_posn===1&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+this.currencyCode+b}else if(this.lc.int_n_sign_posn===1&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===0){return this.lc.negative_sign+b+this.intSep+this.currencyCode}else if(this.lc.int_n_sign_posn===1&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+this.currencyCode+this.intSep+b}else if(this.lc.int_n_sign_posn===1&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===0){return this.lc.negative_sign+this.intSep+b+this.currencyCode}else if(this.lc.int_n_sign_posn===1&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+this.intSep+this.currencyCode+b}else if(this.lc.int_n_sign_posn===2&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===0){return b+this.currencyCode+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===2&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===1){return this.currencyCode+b+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===2&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===0){return b+this.intSep+this.currencyCode+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===2&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===1){return this.currencyCode+this.intSep+b+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===2&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===0){return b+this.currencyCode+this.intSep+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===2&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===1){return this.currencyCode+b+this.intSep+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===3&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===0){return b+this.lc.negative_sign+this.currencyCode}else if(this.lc.int_n_sign_posn===3&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+this.currencyCode+b}else if(this.lc.int_n_sign_posn===3&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===0){return b+this.intSep+this.lc.negative_sign+this.currencyCode}else if(this.lc.int_n_sign_posn===3&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+this.currencyCode+this.intSep+b}else if(this.lc.int_n_sign_posn===3&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===0){return b+this.lc.negative_sign+this.intSep+this.currencyCode}else if(this.lc.int_n_sign_posn===3&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+this.intSep+this.currencyCode+b}else if(this.lc.int_n_sign_posn===4&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===0){return b+this.currencyCode+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===4&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===1){return this.currencyCode+this.lc.negative_sign+b}else if(this.lc.int_n_sign_posn===4&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===0){return b+this.intSep+this.currencyCode+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===4&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===1){return this.currencyCode+this.lc.negative_sign+this.intSep+b}else if(this.lc.int_n_sign_posn===4&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===0){return b+this.currencyCode+this.intSep+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===4&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===1){return this.currencyCode+this.intSep+this.lc.negative_sign+b}}throw"Error: Invalid POSIX LC MONETARY definition"};this._formatAsLocalCurrencyWithNoSym=function(a,b){if(a=="+"){if(this.lc.p_sign_posn===0){return"("+b+")"}else if(this.lc.p_sign_posn===1&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===0){return this.lc.positive_sign+b}else if(this.lc.p_sign_posn===1&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+b}else if(this.lc.p_sign_posn===1&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===0){return this.lc.positive_sign+b}else if(this.lc.p_sign_posn===1&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+b}else if(this.lc.p_sign_posn===1&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===0){return this.lc.positive_sign+" "+b}else if(this.lc.p_sign_posn===1&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+" "+b}else if(this.lc.p_sign_posn===2&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===0){return b+this.lc.positive_sign}else if(this.lc.p_sign_posn===2&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===1){return b+this.lc.positive_sign}else if(this.lc.p_sign_posn===2&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===0){return b+" "+this.lc.positive_sign}else if(this.lc.p_sign_posn===2&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===1){return b+this.lc.positive_sign}else if(this.lc.p_sign_posn===2&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===0){return b+this.lc.positive_sign}else if(this.lc.p_sign_posn===2&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===1){return b+" "+this.lc.positive_sign}else if(this.lc.p_sign_posn===3&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===0){return b+this.lc.positive_sign}else if(this.lc.p_sign_posn===3&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+b}else if(this.lc.p_sign_posn===3&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===0){return b+" "+this.lc.positive_sign}else if(this.lc.p_sign_posn===3&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+" "+b}else if(this.lc.p_sign_posn===3&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===0){return b+this.lc.positive_sign}else if(this.lc.p_sign_posn===3&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+" "+b}else if(this.lc.p_sign_posn===4&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===0){return b+this.lc.positive_sign}else if(this.lc.p_sign_posn===4&&this.lc.p_sep_by_space===0&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+b}else if(this.lc.p_sign_posn===4&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===0){return b+" "+this.lc.positive_sign}else if(this.lc.p_sign_posn===4&&this.lc.p_sep_by_space===1&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+" "+b}else if(this.lc.p_sign_posn===4&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===0){return b+" "+this.lc.positive_sign}else if(this.lc.p_sign_posn===4&&this.lc.p_sep_by_space===2&&this.lc.p_cs_precedes===1){return this.lc.positive_sign+b}}else if(a=="-"){if(this.lc.n_sign_posn===0){return"("+b+")"}else if(this.lc.n_sign_posn===1&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===0){return this.lc.negative_sign+b}else if(this.lc.n_sign_posn===1&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+b}else if(this.lc.n_sign_posn===1&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===0){return this.lc.negative_sign+b}else if(this.lc.n_sign_posn===1&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+" "+b}else if(this.lc.n_sign_posn===1&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===0){return this.lc.negative_sign+" "+b}else if(this.lc.n_sign_posn===1&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+" "+b}else if(this.lc.n_sign_posn===2&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===0){return b+this.lc.negative_sign}else if(this.lc.n_sign_posn===2&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===1){return b+this.lc.negative_sign}else if(this.lc.n_sign_posn===2&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===0){return b+" "+this.lc.negative_sign}else if(this.lc.n_sign_posn===2&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===1){return b+this.lc.negative_sign}else if(this.lc.n_sign_posn===2&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===0){return b+" "+this.lc.negative_sign}else if(this.lc.n_sign_posn===2&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===1){return b+" "+this.lc.negative_sign}else if(this.lc.n_sign_posn===3&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===0){return b+this.lc.negative_sign}else if(this.lc.n_sign_posn===3&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+b}else if(this.lc.n_sign_posn===3&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===0){return b+" "+this.lc.negative_sign}else if(this.lc.n_sign_posn===3&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+" "+b}else if(this.lc.n_sign_posn===3&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===0){return b+this.lc.negative_sign}else if(this.lc.n_sign_posn===3&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+" "+b}else if(this.lc.n_sign_posn===4&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===0){return b+this.lc.negative_sign}else if(this.lc.n_sign_posn===4&&this.lc.n_sep_by_space===0&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+b}else if(this.lc.n_sign_posn===4&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===0){return b+" "+this.lc.negative_sign}else if(this.lc.n_sign_posn===4&&this.lc.n_sep_by_space===1&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+" "+b}else if(this.lc.n_sign_posn===4&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===0){return b+" "+this.lc.negative_sign}else if(this.lc.n_sign_posn===4&&this.lc.n_sep_by_space===2&&this.lc.n_cs_precedes===1){return this.lc.negative_sign+b}}throw"Error: Invalid POSIX LC MONETARY definition"};this._formatAsInternationalCurrencyWithNoSym=function(a,b){if(a=="+"){if(this.lc.int_p_sign_posn===0){return"("+b+")"}else if(this.lc.int_p_sign_posn===1&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===0){return this.lc.positive_sign+b}else if(this.lc.int_p_sign_posn===1&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+b}else if(this.lc.int_p_sign_posn===1&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===0){return this.lc.positive_sign+b}else if(this.lc.int_p_sign_posn===1&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+this.intSep+b}else if(this.lc.int_p_sign_posn===1&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===0){return this.lc.positive_sign+this.intSep+b}else if(this.lc.int_p_sign_posn===1&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+this.intSep+b}else if(this.lc.int_p_sign_posn===2&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===0){return b+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===2&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===1){return b+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===2&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===0){return b+this.intSep+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===2&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===1){return b+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===2&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===0){return b+this.intSep+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===2&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===1){return b+this.intSep+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===3&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===0){return b+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===3&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+b}else if(this.lc.int_p_sign_posn===3&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===0){return b+this.intSep+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===3&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+this.intSep+b}else if(this.lc.int_p_sign_posn===3&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===0){return b+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===3&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+this.intSep+b}else if(this.lc.int_p_sign_posn===4&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===0){return b+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===4&&this.lc.int_p_sep_by_space===0&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+b}else if(this.lc.int_p_sign_posn===4&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===0){return b+this.intSep+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===4&&this.lc.int_p_sep_by_space===1&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+this.intSep+b}else if(this.lc.int_p_sign_posn===4&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===0){return b+this.intSep+this.lc.positive_sign}else if(this.lc.int_p_sign_posn===4&&this.lc.int_p_sep_by_space===2&&this.lc.int_p_cs_precedes===1){return this.lc.positive_sign+b}}else if(a=="-"){if(this.lc.int_n_sign_posn===0){return"("+b+")"}else if(this.lc.int_n_sign_posn===1&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===0){return this.lc.negative_sign+b}else if(this.lc.int_n_sign_posn===1&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+b}else if(this.lc.int_n_sign_posn===1&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===0){return this.lc.negative_sign+b}else if(this.lc.int_n_sign_posn===1&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+this.intSep+b}else if(this.lc.int_n_sign_posn===1&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===0){return this.lc.negative_sign+this.intSep+b}else if(this.lc.int_n_sign_posn===1&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+this.intSep+b}else if(this.lc.int_n_sign_posn===2&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===0){return b+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===2&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===1){return b+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===2&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===0){return b+this.intSep+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===2&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===1){return b+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===2&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===0){return b+this.intSep+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===2&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===1){return b+this.intSep+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===3&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===0){return b+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===3&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+b}else if(this.lc.int_n_sign_posn===3&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===0){return b+this.intSep+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===3&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+this.intSep+b}else if(this.lc.int_n_sign_posn===3&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===0){return b+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===3&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+this.intSep+b}else if(this.lc.int_n_sign_posn===4&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===0){return b+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===4&&this.lc.int_n_sep_by_space===0&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+b}else if(this.lc.int_n_sign_posn===4&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===0){return b+this.intSep+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===4&&this.lc.int_n_sep_by_space===1&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+this.intSep+b}else if(this.lc.int_n_sign_posn===4&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===0){return b+this.intSep+this.lc.negative_sign}else if(this.lc.int_n_sign_posn===4&&this.lc.int_n_sep_by_space===2&&this.lc.int_n_cs_precedes===1){return this.lc.negative_sign+b}}throw"Error: Invalid POSIX LC_MONETARY definition"}};jsworld.NumericParser=function(a){if(typeof a!="object"||a._className!="jsworld.Locale")throw"Constructor error: You must provide a valid jsworld.Locale instance";this.lc=a;this.parse=function(a){if(typeof a!="string")throw"Parse error: Argument must be a string";var b=jsworld._trim(a);b=jsworld._stringReplaceAll(a,this.lc.thousands_sep,"");b=jsworld._stringReplaceAll(b,this.lc.decimal_point,".");if(jsworld._isNumber(b))return parseFloat(b,10);else throw"Parse error: Invalid number string"}};jsworld.DateTimeParser=function(a){if(typeof a!="object"||a._className!="jsworld.Locale")throw"Constructor error: You must provide a valid jsworld.Locale instance.";this.lc=a;this.parseTime=function(a){if(typeof a!="string")throw"Parse error: Argument must be a string";var b=this._extractTokens(this.lc.t_fmt,a);var c=false;if(b.hour!==null&&b.minute!==null&&b.second!==null){c=true}else if(b.hourAmPm!==null&&b.am!==null&&b.minute!==null&&b.second!==null){if(b.am){b.hour=parseInt(b.hourAmPm,10)}else{if(b.hourAmPm==12)b.hour=0;else b.hour=parseInt(b.hourAmPm,10)+12}c=true}if(c)return jsworld._zeroPad(b.hour,2)+":"+jsworld._zeroPad(b.minute,2)+":"+jsworld._zeroPad(b.second,2);else throw"Parse error: Invalid/ambiguous time string"};this.parseDate=function(a){if(typeof a!="string")throw"Parse error: Argument must be a string";var b=this._extractTokens(this.lc.d_fmt,a);var c=false;if(b.year!==null&&b.month!==null&&b.day!==null){c=true}if(c)return jsworld._zeroPad(b.year,4)+"-"+jsworld._zeroPad(b.month,2)+"-"+jsworld._zeroPad(b.day,2);else throw"Parse error: Invalid date string"};this.parseDateTime=function(a){if(typeof a!="string")throw"Parse error: Argument must be a string";var b=this._extractTokens(this.lc.d_t_fmt,a);var c=false;var d=false;if(b.hour!==null&&b.minute!==null&&b.second!==null){c=true}else if(b.hourAmPm!==null&&b.am!==null&&b.minute!==null&&b.second!==null){if(b.am){b.hour=parseInt(b.hourAmPm,10)}else{if(b.hourAmPm==12)b.hour=0;else b.hour=parseInt(b.hourAmPm,10)+12}c=true}if(b.year!==null&&b.month!==null&&b.day!==null){d=true}if(d&&c)return jsworld._zeroPad(b.year,4)+"-"+jsworld._zeroPad(b.month,2)+"-"+jsworld._zeroPad(b.day,2)+" "+jsworld._zeroPad(b.hour,2)+":"+jsworld._zeroPad(b.minute,2)+":"+jsworld._zeroPad(b.second,2);else throw"Parse error: Invalid/ambiguous date/time string"};this._extractTokens=function(a,b){var c={year:null,month:null,day:null,hour:null,hourAmPm:null,am:null,minute:null,second:null,weekday:null};while(a.length>0){if(a.charAt(0)=="%"&&a.charAt(1)!=""){var d=a.substring(0,2);if(d=="%%"){b=b.substring(1)}else if(d=="%a"){for(var e=0;e<this.lc.abday.length;e++){if(jsworld._stringStartsWith(b,this.lc.abday[e])){c.weekday=e;b=b.substring(this.lc.abday[e].length);break}}if(c.weekday===null)throw"Parse error: Unrecognised abbreviated weekday name (%a)"}else if(d=="%A"){for(var e=0;e<this.lc.day.length;e++){if(jsworld._stringStartsWith(b,this.lc.day[e])){c.weekday=e;b=b.substring(this.lc.day[e].length);break}}if(c.weekday===null)throw"Parse error: Unrecognised weekday name (%A)"}else if(d=="%b"||d=="%h"){for(var e=0;e<this.lc.abmon.length;e++){if(jsworld._stringStartsWith(b,this.lc.abmon[e])){c.month=e+1;b=b.substring(this.lc.abmon[e].length);break}}if(c.month===null)throw"Parse error: Unrecognised abbreviated month name (%b)"}else if(d=="%B"){for(var e=0;e<this.lc.mon.length;e++){if(jsworld._stringStartsWith(b,this.lc.mon[e])){c.month=e+1;b=b.substring(this.lc.mon[e].length);break}}if(c.month===null)throw"Parse error: Unrecognised month name (%B)"}else if(d=="%d"){if(/^0[1-9]|[1-2][0-9]|3[0-1]/.test(b)){c.day=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised day of the month (%d)"}else if(d=="%e"){var f=b.match(/^\s?(\d{1,2})/);c.day=parseInt(f,10);if(isNaN(c.day)||c.day<1||c.day>31)throw"Parse error: Unrecognised day of the month (%e)";b=b.substring(f.length)}else if(d=="%F"){if(/^\d\d\d\d/.test(b)){c.year=parseInt(b.substring(0,4),10);b=b.substring(4)}else{throw"Parse error: Unrecognised date (%F)"}if(jsworld._stringStartsWith(b,"-"))b=b.substring(1);else throw"Parse error: Unrecognised date (%F)";if(/^0[1-9]|1[0-2]/.test(b)){c.month=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised date (%F)";if(jsworld._stringStartsWith(b,"-"))b=b.substring(1);else throw"Parse error: Unrecognised date (%F)";if(/^0[1-9]|[1-2][0-9]|3[0-1]/.test(b)){c.day=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised date (%F)"}else if(d=="%H"){if(/^[0-1][0-9]|2[0-3]/.test(b)){c.hour=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised hour (%H)"}else if(d=="%I"){if(/^0[1-9]|1[0-2]/.test(b)){c.hourAmPm=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised hour (%I)"}else if(d=="%k"){var g=b.match(/^(\d{1,2})/);c.hour=parseInt(g,10);if(isNaN(c.hour)||c.hour<0||c.hour>23)throw"Parse error: Unrecognised hour (%k)";b=b.substring(g.length)}else if(d=="%l"){var g=b.match(/^(\d{1,2})/);c.hourAmPm=parseInt(g,10);if(isNaN(c.hourAmPm)||c.hourAmPm<1||c.hourAmPm>12)throw"Parse error: Unrecognised hour (%l)";b=b.substring(g.length)}else if(d=="%m"){if(/^0[1-9]|1[0-2]/.test(b)){c.month=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised month (%m)"}else if(d=="%M"){if(/^[0-5][0-9]/.test(b)){c.minute=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised minute (%M)"}else if(d=="%n"){if(b.charAt(0)=="\n")b=b.substring(1);else throw"Parse error: Unrecognised new line (%n)"}else if(d=="%p"){if(jsworld._stringStartsWith(b,this.lc.am)){c.am=true;b=b.substring(this.lc.am.length)}else if(jsworld._stringStartsWith(b,this.lc.pm)){c.am=false;b=b.substring(this.lc.pm.length)}else throw"Parse error: Unrecognised AM/PM value (%p)"}else if(d=="%P"){if(jsworld._stringStartsWith(b,this.lc.am.toLowerCase())){c.am=true;b=b.substring(this.lc.am.length)}else if(jsworld._stringStartsWith(b,this.lc.pm.toLowerCase())){c.am=false;b=b.substring(this.lc.pm.length)}else throw"Parse error: Unrecognised AM/PM value (%P)"}else if(d=="%R"){if(/^[0-1][0-9]|2[0-3]/.test(b)){c.hour=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised time (%R)";if(jsworld._stringStartsWith(b,":"))b=b.substring(1);else throw"Parse error: Unrecognised time (%R)";if(/^[0-5][0-9]/.test(b)){c.minute=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised time (%R)"}else if(d=="%S"){if(/^[0-5][0-9]/.test(b)){c.second=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised second (%S)"}else if(d=="%T"){if(/^[0-1][0-9]|2[0-3]/.test(b)){c.hour=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised time (%T)";if(jsworld._stringStartsWith(b,":"))b=b.substring(1);else throw"Parse error: Unrecognised time (%T)";if(/^[0-5][0-9]/.test(b)){c.minute=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised time (%T)";if(jsworld._stringStartsWith(b,":"))b=b.substring(1);else throw"Parse error: Unrecognised time (%T)";if(/^[0-5][0-9]/.test(b)){c.second=parseInt(b.substring(0,2),10);b=b.substring(2)}else throw"Parse error: Unrecognised time (%T)"}else if(d=="%w"){if(/^\d/.test(b)){c.weekday=parseInt(b.substring(0,1),10);b=b.substring(1)}else throw"Parse error: Unrecognised weekday number (%w)"}else if(d=="%y"){if(/^\d\d/.test(b)){var h=parseInt(b.substring(0,2),10);if(h>50)c.year=1900+h;else c.year=2e3+h;b=b.substring(2)}else throw"Parse error: Unrecognised year (%y)"}else if(d=="%Y"){if(/^\d\d\d\d/.test(b)){c.year=parseInt(b.substring(0,4),10);b=b.substring(4)}else throw"Parse error: Unrecognised year (%Y)"}else if(d=="%Z"){if(a.length===0)break}a=a.substring(2)}else{if(a.charAt(0)!=b.charAt(0))throw'Parse error: Unexpected symbol "'+b.charAt(0)+'" in date/time string';a=a.substring(1);b=b.substring(1)}}return c}};jsworld.MonetaryParser=function(a){if(typeof a!="object"||a._className!="jsworld.Locale")throw"Constructor error: You must provide a valid jsworld.Locale instance";this.lc=a;this.parse=function(a){if(typeof a!="string")throw"Parse error: Argument must be a string";var b=this._detectCurrencySymbolType(a);var c,d;if(b=="local"){c="local";d=a.replace(this.lc.getCurrencySymbol(),"")}else if(b=="int"){c="int";d=a.replace(this.lc.getIntCurrencySymbol(),"")}else if(b=="none"){c="local";d=a}else throw"Parse error: Internal assert failure";d=jsworld._stringReplaceAll(d,this.lc.mon_thousands_sep,"");d=d.replace(this.lc.mon_decimal_point,".");d=d.replace(/\s*/g,"");d=this._removeLocalNonNegativeSign(d,c);d=this._normaliseNegativeSign(d,c);if(jsworld._isNumber(d))return parseFloat(d,10);else throw"Parse error: Invalid currency amount string"};this._detectCurrencySymbolType=function(a){if(this.lc.getCurrencySymbol().length>this.lc.getIntCurrencySymbol().length){if(a.indexOf(this.lc.getCurrencySymbol())!=-1)return"local";else if(a.indexOf(this.lc.getIntCurrencySymbol())!=-1)return"int";else return"none"}else{if(a.indexOf(this.lc.getIntCurrencySymbol())!=-1)return"int";else if(a.indexOf(this.lc.getCurrencySymbol())!=-1)return"local";else return"none"}};this._removeLocalNonNegativeSign=function(a,b){a=a.replace(this.lc.positive_sign,"");if((b=="local"&&this.lc.p_sign_posn===0||b=="int"&&this.lc.int_p_sign_posn===0)&&/\(\d+\.?\d*\)/.test(a)){a=a.replace("(","");a=a.replace(")","")}return a};this._normaliseNegativeSign=function(a,b){a=a.replace(this.lc.negative_sign,"-");if(b=="local"&&this.lc.n_sign_posn===0||b=="int"&&this.lc.int_n_sign_posn===0){if(/^\(\d+\.?\d*\)$/.test(a)){a=a.replace("(","");a=a.replace(")","");return"-"+a}}if(b=="local"&&this.lc.n_sign_posn==2||b=="int"&&this.lc.int_n_sign_posn==2){if(/^\d+\.?\d*-$/.test(a)){a=a.replace("-","");return"-"+a}}if(b=="local"&&this.lc.n_cs_precedes===0&&this.lc.n_sign_posn==3||b=="local"&&this.lc.n_cs_precedes===0&&this.lc.n_sign_posn==4||b=="int"&&this.lc.int_n_cs_precedes===0&&this.lc.int_n_sign_posn==3||b=="int"&&this.lc.int_n_cs_precedes===0&&this.lc.int_n_sign_posn==4){if(/^\d+\.?\d*-$/.test(a)){a=a.replace("-","");return"-"+a}}return a}}


if(typeof POSIX_LC == "undefined") var POSIX_LC = {};

POSIX_LC.en_US = {
    "decimal_point"      : ".",
    "thousands_sep"      : ",",
    "grouping"           : "3",
    "abday"              : ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],
    "day"                : ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"],
    "abmon"              : ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    "mon"                : ["January","February","March","April","May","June","July","August","September","October","November","December"],
    "d_fmt"              : "%m/%e/%y",
    "t_fmt"              : "%I:%M:%S %p",
    "d_t_fmt"            : "%B %e, %Y %I:%M:%S %p %Z",
    "am_pm"              : ["AM","PM"],
    "int_curr_symbol"    : "USD ",
    "currency_symbol"    : "\u0024",
    "mon_decimal_point"  : ".",
    "mon_thousands_sep"  : ",",
    "mon_grouping"       : "3",
    "positive_sign"      : "",
    "negative_sign"      : "-",
    "int_frac_digits"    : 2,
    "frac_digits"        : 2,
    "p_cs_precedes"      : 1,
    "n_cs_precedes"      : 1,
    "p_sep_by_space"     : 0,
    "n_sep_by_space"     : 0,
    "p_sign_posn"        : 1,
    "n_sign_posn"        : 1,
    "int_p_cs_precedes"  : 1,
    "int_n_cs_precedes"  : 1,
    "int_p_sep_by_space" : 0,
    "int_n_sep_by_space" : 0,
    "int_p_sign_posn"    : 1,
    "int_n_sign_posn"    : 1
}

if(typeof POSIX_LC == "undefined") var POSIX_LC = {};

POSIX_LC.fr_FR = {
    "decimal_point"      : ",",
    "thousands_sep"      : "\u00a0",
    "grouping"           : "3",
    "abday"              : ["dim.","lun.","mar.",
                            "mer.","jeu.","ven.",
                "sam."],
    "day"                : ["dimanche","lundi","mardi",
                            "mercredi","jeudi","vendredi",
                "samedi"],
    "abmon"              : ["janv.","f\u00e9vr.","mars",
                            "avr.","mai","juin",
                "juil.","ao\u00fbt","sept.",
                "oct.","nov.","d\u00e9c."],
    "mon"                : ["janvier","f\u00e9vrier","mars",
                            "avril","mai","juin",
                "juillet","ao\u00fbt","septembre",
                "octobre","novembre","d\u00e9cembre"],
    "d_fmt"              : "%d/%m/%y",
    "t_fmt"              : "%H:%M:%S",
    "d_t_fmt"            : "%e %B %Y %H:%M:%S %Z",
    "am_pm"              : ["AM","PM"],
    "int_curr_symbol"    : "EUR ",
    "currency_symbol"    : "\u20ac",
    "mon_decimal_point"  : ",",
    "mon_thousands_sep"  : "\u00a0",
    "mon_grouping"       : "3",
    "positive_sign"      : "",
    "negative_sign"      : "-",
    "int_frac_digits"    : 2,
    "frac_digits"        : 2,
    "p_cs_precedes"      : 0,
    "n_cs_precedes"      : 0,
    "p_sep_by_space"     : 1,
    "n_sep_by_space"     : 1,
    "p_sign_posn"        : 1,
    "n_sign_posn"        : 1,
    "int_p_cs_precedes"  : 0,
    "int_n_cs_precedes"  : 0,
    "int_p_sep_by_space" : 1,
    "int_n_sep_by_space" : 1,
    "int_p_sign_posn"    : 1,
    "int_n_sign_posn"    : 1
};

/** https://github.com/csnover/js-iso8601 */(function(n,f){var u=n.parse,c=[1,4,5,6,7,10,11];n.parse=function(t){var i,o,a=0;if(o=/^(\d{4}|[+\-]\d{6})(?:-(\d{2})(?:-(\d{2}))?)?(?:T(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{3}))?)?(?:(Z)|([+\-])(\d{2})(?::(\d{2}))?)?)?$/.exec(t)){for(var v=0,r;r=c[v];++v)o[r]=+o[r]||0;o[2]=(+o[2]||1)-1,o[3]=+o[3]||1,o[8]!=="Z"&&o[9]!==f&&(a=o[10]*60+o[11],o[9]==="+"&&(a=0-a)),i=n.UTC(o[1],o[2],o[3],o[4],o[5]+a,o[6],o[7])}else i=u?u(t):NaN;return i}})(Date)

/*!
 * geo-location-javascript v0.4.3
 * http://code.google.com/p/geo-location-javascript/
 *
 * Copyright (c) 2009 Stan Wiechers
 * Licensed under the MIT licenses.
 *
 * Revision: $Rev: 68 $: 
 * Author: $Author: whoisstan $:
 * Date: $Date: 2010-02-15 13:42:19 +0100 (Mon, 15 Feb 2010) $:    
 */
var geo_position_js=function() {

        var pub = {};
        var provider=null;

        pub.getCurrentPosition = function(successCallback,errorCallback,options)
        {
                provider.getCurrentPosition(successCallback, errorCallback,options);
        }

        pub.init = function()
        {			
                try
                {
                        if (typeof(geo_position_js_simulator)!="undefined")
                        {
                                provider=geo_position_js_simulator;
                        }
                        else if (typeof(bondi)!="undefined" && typeof(bondi.geolocation)!="undefined")
                        {
                                provider=bondi.geolocation;
                        }
                        else if (typeof(navigator.geolocation)!="undefined")
                        {
                                provider=navigator.geolocation;
                                pub.getCurrentPosition = function(successCallback, errorCallback, options)
                                {
                                        function _successCallback(p)
                                        {
                                                //for mozilla geode,it returns the coordinates slightly differently
                                                if(typeof(p.latitude)!="undefined")
                                                {
                                                        successCallback({timestamp:p.timestamp, coords: {latitude:p.latitude,longitude:p.longitude}});
                                                }
                                                else
                                                {
                                                        successCallback(p);
                                                }
                                        }
                                        provider.getCurrentPosition(_successCallback,errorCallback,options);
                                }
                        }
                         else if(typeof(window.google)!="undefined" && typeof(google.gears)!="undefined")
                        {
                                provider=google.gears.factory.create('beta.geolocation');
                        }
                        else if ( typeof(Mojo) !="undefined" && typeof(Mojo.Service.Request)!="Mojo.Service.Request")
                        {
                                provider=true;
                                pub.getCurrentPosition = function(successCallback, errorCallback, options)
                                {

                                parameters={};
                                if(options)
                                {
                                         //http://developer.palm.com/index.php?option=com_content&view=article&id=1673#GPS-getCurrentPosition
                                         if (options.enableHighAccuracy && options.enableHighAccuracy==true)
                                         {
                                                parameters.accuracy=1;
                                         }
                                         if (options.maximumAge)
                                         {
                                                parameters.maximumAge=options.maximumAge;
                                         }
                                         if (options.responseTime)
                                         {
                                                if(options.responseTime<5)
                                                {
                                                        parameters.responseTime=1;
                                                }
                                                else if (options.responseTime<20)
                                                {
                                                        parameters.responseTime=2;
                                                }
                                                else
                                                {
                                                        parameters.timeout=3;
                                                }
                                         }
                                }


                                 r=new Mojo.Service.Request('palm://com.palm.location', {
                                        method:"getCurrentPosition",
                                            parameters:parameters,
                                            onSuccess: function(p){successCallback({timestamp:p.timestamp, coords: {latitude:p.latitude, longitude:p.longitude,heading:p.heading}});},
                                            onFailure: function(e){
                                                                if (e.errorCode==1)
                                                                {
                                                                        errorCallback({code:3,message:"Timeout"});
                                                                }
                                                                else if (e.errorCode==2)
                                                                {
                                                                        errorCallback({code:2,message:"Position Unavailable"});
                                                                }
                                                                else
                                                                {
                                                                        errorCallback({code:0,message:"Unknown Error: webOS-code"+errorCode});
                                                                }
                                                        }
                                            });
                                }

                        }
                        else if (typeof(device)!="undefined" && typeof(device.getServiceObject)!="undefined")
                        {
                                provider=device.getServiceObject("Service.Location", "ILocation");

                                //override default method implementation
                                pub.getCurrentPosition = function(successCallback, errorCallback, options)
                                {
                                        function callback(transId, eventCode, result) {
                                            if (eventCode == 4)
                                                {
                                                errorCallback({message:"Position unavailable", code:2});
                                            }
                                                else
                                                {
                                                        //no timestamp of location given?
                                                        successCallback({timestamp:null, coords: {latitude:result.ReturnValue.Latitude, longitude:result.ReturnValue.Longitude, altitude:result.ReturnValue.Altitude,heading:result.ReturnValue.Heading}});
                                                }
                                        }
                                        //location criteria
                                    var criteria = new Object();
                                criteria.LocationInformationClass = "BasicLocationInformation";
                                        //make the call
                                        provider.ILocation.GetLocation(criteria,callback);
                                }
                        }
                }
                catch (e){ 
					alert("error="+e);
					if(typeof(console)!="undefined")
					{
						console.log(e);
					}
					return false;
				}
                return  provider!=null;
        }


        return pub;
}();
// Couldn't get unminified version to work , go here for docs => https://github.com/iamnoah/writeCapture
(function(E,a){var j=a.document;function A(Q){var Z=j.createElement("div");j.body.insertBefore(Z,null);E.replaceWith(Z,'<script type="text/javascript">'+Q+"<\/script>")}E=E||(function(Q){return{ajax:Q.ajax,$:function(Z){return Q(Z)[0]},replaceWith:function(Z,ad){var ac=Q(Z)[0];var ab=ac.nextSibling,aa=ac.parentNode;Q(ac).remove();if(ab){Q(ab).before(ad)}else{Q(aa).append(ad)}},onLoad:function(Z){Q(Z)},copyAttrs:function(af,ab){var ad=Q(ab),aa=af.attributes;for(var ac=0,Z=aa.length;ac<Z;ac++){if(aa[ac]&&aa[ac].value){try{ad.attr(aa[ac].name,aa[ac].value)}catch(ae){}}}}}})(a.jQuery);E.copyAttrs=E.copyAttrs||function(){};E.onLoad=E.onLoad||function(){throw"error: autoAsync cannot be used without jQuery or defining writeCaptureSupport.onLoad"};function P(ab,aa){for(var Z=0,Q=ab.length;Z<Q;Z++){if(aa(ab[Z])===false){return}}}function v(Q){return Object.prototype.toString.call(Q)==="[object Function]"}function p(Q){return Object.prototype.toString.call(Q)==="[object String]"}function u(aa,Z,Q){return Array.prototype.slice.call(aa,Z||0,Q||aa&&aa.length)}function D(ab,aa){var Q=false;P(ab,Z);function Z(ac){return !(Q=aa(ac))}return Q}function L(Q){this._queue=[];this._children=[];this._parent=Q;if(Q){Q._addChild(this)}}L.prototype={_addChild:function(Q){this._children.push(Q)},push:function(Q){this._queue.push(Q);this._bubble("_doRun")},pause:function(){this._bubble("_doPause")},resume:function(){this._bubble("_doResume")},_bubble:function(Z){var Q=this;while(!Q[Z]){Q=Q._parent}return Q[Z]()},_next:function(){if(D(this._children,Q)){return true}function Q(aa){return aa._next()}var Z=this._queue.shift();if(Z){Z()}return !!Z}};function i(Q){if(Q){return new L(Q)}L.call(this);this.paused=0}i.prototype=(function(){function Q(){}Q.prototype=L.prototype;return new Q()})();i.prototype._doRun=function(){if(!this.running){this.running=true;try{while(this.paused<1&&this._next()){}}finally{this.running=false}}};i.prototype._doPause=function(){this.paused++};i.prototype._doResume=function(){this.paused--;this._doRun()};function M(){}M.prototype={_html:"",open:function(){this._opened=true;if(this._delegate){this._delegate.open()}},write:function(Q){if(this._closed){return}this._written=true;if(this._delegate){this._delegate.write(Q)}else{this._html+=Q}},writeln:function(Q){this.write(Q+"\n")},close:function(){this._closed=true;if(this._delegate){this._delegate.close()}},copyTo:function(Q){this._delegate=Q;Q.foobar=true;if(this._opened){Q.open()}if(this._written){Q.write(this._html)}if(this._closed){Q.close()}}};var e=(function(){var Q={f:j.getElementById};try{Q.f.call(j,"abc");return true}catch(Z){return false}})();function I(Q){P(Q,function(Z){var aa=j.getElementById(Z.id);if(!aa){l("<proxyGetElementById - finish>","no element in writen markup with id "+Z.id);return}P(Z.el.childNodes,function(ab){aa.appendChild(ab)});if(aa.contentWindow){a.setTimeout(function(){Z.el.contentWindow.document.copyTo(aa.contentWindow.document)},1)}E.copyAttrs(Z.el,aa)})}function s(Z,Q){if(Q&&Q[Z]===false){return false}return Q&&Q[Z]||o[Z]}function x(Z,ai){var ae=[],ad=s("proxyGetElementById",ai),ag=s("writeOnGetElementById",ai),Q={write:j.write,writeln:j.writeln,finish:function(){},out:""};Z.state=Q;j.write=ah;j.writeln=aa;if(ad||ag){Q.getEl=j.getElementById;j.getElementById=ab;if(ag){findEl=af}else{findEl=ac;Q.finish=function(){I(ae)}}}function ah(aj){Q.out+=aj}function aa(aj){Q.out+=aj+"\n"}function ac(ak){var aj=j.createElement("div");ae.push({id:ak,el:aj});aj.contentWindow={document:new M()};return aj}function af(al){var aj=E.$(Z.target);var ak=j.createElement("div");aj.parentNode.insertBefore(ak,aj);E.replaceWith(ak,Q.out);Q.out="";return e?Q.getEl.call(j,al):Q.getEl(al)}function ab(ak){var aj=e?Q.getEl.call(j,ak):Q.getEl(ak);return aj||findEl(ak)}return Q}function V(Q){j.write=Q.write;j.writeln=Q.writeln;if(Q.getEl){j.getElementById=Q.getEl}return Q.out}function N(Q){return Q&&Q.replace(/^\s*<!(\[CDATA\[|--)/,"").replace(/(\]\]|--)>\s*$/,"")}function b(){}function d(Z,Q){console.error("Error",Q,"executing code:",Z)}var l=v(a.console&&console.error)?d:b;function S(aa,Z,Q){var ab=x(Z,Q);try{A(N(aa))}catch(ac){l(aa,ac)}finally{V(ab)}return ab}function O(Z){var Q=/^(\w+:)?\/\/([^\/?#]+)/.exec(Z);return Q&&(Q[1]&&Q[1]!=location.protocol||Q[2]!=location.host)}function T(Q){return new RegExp(Q+"=(?:([\"'])([\\s\\S]*?)\\1|([^\\s>]+))","i")}function k(Q){var Z=T(Q);return function(aa){var ab=Z.exec(aa)||[];return ab[2]||ab[3]}}var r=/(<script[\s\S]*?>)([\s\S]*?)<\/script>/ig,n=T("src"),X=k("src"),q=k("type"),Y=k("language"),C="__document_write_ajax_callbacks__",B="__document_write_ajax_div-",g="window['"+C+"']['%d']();",m=a[C]={},w='<script type="text/javascript">'+g+"<\/script>",H=0;function c(){return(++H).toString()}function G(Z,aa){var Q;if(v(Z)){Q=Z;Z=null}Z=Z||{};Q=Q||Z&&Z.done;Z.done=aa?function(){aa(Q)}:Q;return Z}var z=new i();var y=[];var f=window._debugWriteCapture?function(){}:function(Q,aa,Z){y.push({type:Q,src:aa,data:Z})};var K=window._debugWriteCapture?function(){}:function(){y.push(arguments)};function W(Q){var Z=c();m[Z]=function(){Q();delete m[Z]};return Z}function J(Q){return w.replace(/%d/,W(Q))}function R(ac,ag,aa,ae){var ad=aa&&new i(aa)||z;ag=G(ag);var ab=s("done",ag);var Q="";var Z=s("fixUrls",ag);if(!v(Z)){Z=function(ah){return ah}}if(v(ab)){Q=J(function(){ad.push(ab)})}return ac.replace(r,af)+Q;function af(aj,av,ai){var an=X(av),am=q(av)||"",aB=Y(av)||"",aA=(!am&&!aB)||am.toLowerCase().indexOf("javascript")!==-1||aB.toLowerCase().indexOf("javascript")!==-1;f("replace",an,aj);if(!aA){return aj}var aw=W(ap),ao=B+aw,au,al={target:"#"+ao,parent:ae};function ap(){ad.push(au)}if(an){an=Z(an);av=av.replace(n,"");if(O(an)){au=az}else{if(s("asyncAll",ag)){au=ay()}else{au=at}}}else{au=ax}function ax(){ah(ai)}function at(){E.ajax({url:an,type:"GET",dataType:"text",async:false,success:function(aC){ah(aC)}})}function ak(aE,aC,aD){l("<XHR for "+an+">",aD);ad.resume()}function aq(){return J(function(){ad.resume()})}function ay(){var aE,aD;function aC(aG,aF){if(!aE){aD=aG;return}try{ah(aG,aq())}catch(aH){l(aG,aH)}}E.ajax({url:an,type:"GET",dataType:"text",async:true,success:aC,error:ak});return function(){aE=true;if(aD){ah(aD)}else{ad.pause()}}}function az(aC){var aE=x(al,ag);ad.pause();f("pause",an);E.ajax({url:an,type:"GET",dataType:"script",success:aD,error:ak});function aD(aH,aG,aF){f("out",an,aE.out);ar(V(aE),J(aE.finish)+aq());f("resume",an)}}function ah(aD,aC){var aE=S(aD,al,ag);aC=J(aE.finish)+(aC||"");ar(aE.out,aC)}function ar(aD,aC){E.replaceWith(al.target,R(aD,null,ad,al)+(aC||""))}return'<div style="display: none" id="'+ao+'"></div>'+av+g.replace(/%d/,aw)+"<\/script>"}}function F(Z,aa){var Q=z;P(Z,function(ab){Q.push(ac);function ac(){ab.action(R(ab.html,ab.options,Q),ab)}});if(aa){Q.push(aa)}}function U(Q){var Z=Q;while(Z&&Z.nodeType===1){Q=Z;Z=Z.lastChild;while(Z&&Z.nodeType!==1){Z=Z.previousSibling}}return Q}function h(Q){var aa=j.write,ad=j.writeln,Z,ab=[];j.writeln=function(ae){j.write(ae+"\n")};var ac;j.write=function(af){var ae=U(j.body);if(ae!==Z){Z=ae;ab.push(ac={el:ae,out:[]})}ac.out.push(af)};E.onLoad(function(){var ah,ak,af,aj,ai;Q=G(Q);ai=Q.done;Q.done=function(){j.write=aa;j.writeln=ad;if(ai){ai()}};for(var ag=0,ae=ab.length;ag<ae;ag++){ah=ab[ag].el;ak=j.createElement("div");ah.parentNode.insertBefore(ak,ah.nextSibling);af=ab[ag].out.join("");aj=ae-ag===1?R(af,Q):R(af);E.replaceWith(ak,aj)}})}var t="writeCapture";var o=a[t]={_original:a[t],fixUrls:function(Q){return Q.replace(/&amp;/g,"&")},noConflict:function(){a[t]=this._original;return this},debug:y,proxyGetElementById:false,_forTest:{Q:i,GLOBAL_Q:z,$:E,matchAttr:k,slice:u,capture:x,uncapture:V,captureWrite:S},replaceWith:function(Q,aa,Z){E.replaceWith(Q,R(aa,Z))},html:function(Q,ab,Z){var aa=E.$(Q);aa.innerHTML="<span/>";E.replaceWith(aa.firstChild,R(ab,Z))},load:function(Q,aa,Z){E.ajax({url:aa,dataType:"text",type:"GET",success:function(ab){o.html(Q,ab,Z)}})},autoAsync:h,sanitize:R,sanitizeSerial:F}})(this.writeCaptureSupport,this);(function(g,d,n){var c={html:h};g.each(["append","prepend","after","before","wrap","wrapAll","replaceWith","wrapInner"],function(){c[this]=i(this)});function a(q){return Object.prototype.toString.call(q)=="[object String]"}function p(u,t,s,r){if(arguments.length==0){return o.call(this)}var q=c[u];if(u=="load"){return l.call(this,t,s,r)}if(!q){j(u)}return b.call(this,t,s,q)}g.fn.writeCapture=p;var k="__writeCaptureJsProxied-fghebd__";function o(){if(this[k]){return this}var r=this;function q(){var t=this,s=false;this[k]=true;g.each(c,function(v){var u=r[v];if(!u){return}t[v]=function(y,x,w){if(!s&&a(y)){try{s=true;return p.call(t,v,y,x,w)}finally{s=false}}return u.apply(t,arguments)}});this.pushStack=function(){return o.call(r.pushStack.apply(t,arguments))};this.endCapture=function(){return r}}q.prototype=r;return new q()}function b(t,s,u){var q,r=this;if(s&&s.done){q=s.done;delete s.done}else{if(g.isFunction(s)){q=s;s=null}}d.sanitizeSerial(g.map(this,function(v){return{html:t,options:s,action:function(w){u.call(v,w)}}}),q&&function(){q.call(r)}||q);return this}function h(q){g(this).html(q)}function i(q){return function(r){g(this)[q](r)}}function l(t,s,v){var r=this,q,u=t.indexOf(" ");if(u>=0){q=t.slice(u,t.length);t=t.slice(0,u)}if(g.isFunction(v)){s=s||{};s.done=v}return g.ajax({url:t,type:s&&s.type||"GET",dataType:"html",data:s&&s.params,complete:f(r,s,q)})}function f(r,s,q){return function(u,t){if(t=="success"||t=="notmodified"){var v=m(u.responseText,q);b.call(r,v,s,h)}}}var e=/jquery-writeCapture-script-placeholder-(\d+)-wc/g;function m(s,r){if(!r||!s){return s}var t=0,q={};return g("<div/>").append(s.replace(/<script(.|\s)*?\/script>/g,function(u){q[t]=u;return"jquery-writeCapture-script-placeholder-"+(t++)+"-wc"})).find(r).html().replace(e,function(u,v){return q[v]})}function j(q){throw"invalid method parameter "+q}g.writeCapture=d})(jQuery,writeCapture.noConflict());

/*!
 * Amplify Store - Persistent Client-Side Storage 1.0.0
 * 
 * Copyright 2011 appendTo LLC. (http://appendto.com/team)
 * Dual licensed under the MIT or GPL licenses.
 * http://appendto.com/open-source-licenses
 * 
 * http://amplifyjs.com
 */
(function( amplify, undefined ) {

var store = amplify.store = function( key, value, options, type ) {
	var type = store.type;
	if ( options && options.type && options.type in store.types ) {
		type = options.type;
	}
	return store.types[ type ]( key, value, options || {} );
};

store.types = {};
store.type = null;
store.addType = function( type, storage ) {
	if ( !store.type ) {
		store.type = type;
	}

	store.types[ type ] = storage;
	store[ type ] = function( key, value, options ) {
		options = options || {};
		options.type = type;
		return store( key, value, options );
	};
}
store.error = function() {
	return "amplify.store quota exceeded"; 
};

var rprefix = /^__amplify__/;
function createFromStorageInterface( storageType, storage ) {
	store.addType( storageType, function( key, value, options ) {
		var storedValue, parsed, i, remove,
			ret = value,
			now = (new Date()).getTime();

		if ( !key ) {
			ret = {};
			remove = [];
			i = 0;
			try {
				// accessing the length property works around a localStorage bug
				// in Firefox 4.0 where the keys don't update cross-page
				// we assign to key just to avoid Closure Compiler from removing
				// the access as "useless code"
				// https://bugzilla.mozilla.org/show_bug.cgi?id=662511
				key = storage.length;

				while ( key = storage.key( i++ ) ) {
					if ( rprefix.test( key ) ) {
						parsed = JSON.parse( storage.getItem( key ) );
						if ( parsed.expires && parsed.expires <= now ) {
							remove.push( key );
						} else {
							ret[ key.replace( rprefix, "" ) ] = parsed.data;
						}
					}
				}
				while ( key = remove.pop() ) {
					storage.removeItem( key );
				}
			} catch ( error ) {}
			return ret;
		}

		// protect against name collisions with direct storage
		key = "__amplify__" + key;

		if ( value === undefined ) {
			storedValue = storage.getItem( key );
			parsed = storedValue ? JSON.parse( storedValue ) : { expires: -1 };
			if ( parsed.expires && parsed.expires <= now ) {
				storage.removeItem( key );
			} else {
				return parsed.data;
			}
		} else {
			if ( value === null ) {
				storage.removeItem( key );
			} else {
				parsed = JSON.stringify({
					data: value,
					expires: options.expires ? now + options.expires : null
				});
				try {
					storage.setItem( key, parsed );
				// quota exceeded
				} catch( error ) {
					// expire old data and try again
					store[ storageType ]();
					try {
						storage.setItem( key, parsed );
					} catch( error ) {
						throw store.error();
					}
				}
			}
		}

		return ret;
	});
}

// localStorage + sessionStorage
// IE 8+, Firefox 3.5+, Safari 4+, Chrome 4+, Opera 10.5+, iPhone 2+, Android 2+
for ( var webStorageType in { localStorage: 1, sessionStorage: 1 } ) {
	// try/catch for file protocol in Firefox
	try {
		if ( window[ webStorageType ].getItem ) {
			createFromStorageInterface( webStorageType, window[ webStorageType ] );
		}
	} catch( e ) {}
}

// globalStorage
// non-standard: Firefox 2+
// https://developer.mozilla.org/en/dom/storage#globalStorage
if ( window.globalStorage ) {
	// try/catch for file protocol in Firefox
	try {
		createFromStorageInterface( "globalStorage",
			window.globalStorage[ window.location.hostname ] );
		// Firefox 2.0 and 3.0 have sessionStorage and globalStorage
		// make sure we default to globalStorage
		// but don't default to globalStorage in 3.5+ which also has localStorage
		if ( store.type === "sessionStorage" ) {
			store.type = "globalStorage";
		}
	} catch( e ) {}
}

// userData
// non-standard: IE 5+
// http://msdn.microsoft.com/en-us/library/ms531424(v=vs.85).aspx
(function() {
	// IE 9 has quirks in userData that are a huge pain
	// rather than finding a way to detect these quirks
	// we just don't register userData if we have localStorage
	if ( store.types.localStorage ) {
		return;
	}

	// append to html instead of body so we can do this from the head
	var div = document.createElement( "div" ),
		attrKey = "amplify";
	div.style.display = "none";
	document.getElementsByTagName( "head" )[ 0 ].appendChild( div );
	if ( div.addBehavior ) {
		div.addBehavior( "#default#userdata" );

		store.addType( "userData", function( key, value, options ) {
			div.load( attrKey );
			var attr, parsed, prevValue, i, remove,
				ret = value,
				now = (new Date()).getTime();

			if ( !key ) {
				ret = {};
				remove = [];
				i = 0;
				while ( attr = div.XMLDocument.documentElement.attributes[ i++ ] ) {
					parsed = JSON.parse( attr.value );
					if ( parsed.expires && parsed.expires <= now ) {
						remove.push( attr.name );
					} else {
						ret[ attr.name ] = parsed.data;
					}
				}
				while ( key = remove.pop() ) {
					div.removeAttribute( key );
				}
				div.save( attrKey );
				return ret;
			}

			// convert invalid characters to dashes
			// http://www.w3.org/TR/REC-xml/#NT-Name
			// simplified to assume the starting character is valid
			// also removed colon as it is invalid in HTML attribute names
			//key = key.replace( /[^-._0-9A-Za-z\xb7\xc0-\xd6\xd8-\xf6\xf8-\u037d\u37f-\u1fff\u200c-\u200d\u203f\u2040\u2070-\u218f]/g, "-" );

			if ( value === undefined ) {
				attr = div.getAttribute( key );
				parsed = attr ? JSON.parse( attr ) : { expires: -1 };
				if ( parsed.expires && parsed.expires <= now ) {
					div.removeAttribute( key );
				} else {
					return parsed.data;
				}
			} else {
				if ( value === null ) {
					div.removeAttribute( key );
				} else {
					// we need to get the previous value in case we need to rollback
					prevValue = div.getAttribute( key );
					parsed = JSON.stringify({
						data: value,
						expires: (options.expires ? (now + options.expires) : null)
					});
					div.setAttribute( key, parsed );
				}
			}

			try {
				div.save( attrKey );
			// quota exceeded
			} catch ( error ) {
				// roll the value back to the previous value
				if ( prevValue === null ) {
					div.removeAttribute( key );
				} else {
					div.setAttribute( key, prevValue );
				}

				// expire old data and try again
				store.userData();
				try {
					div.setAttribute( key, parsed );
					div.save( attrKey );
				} catch ( error ) {
					// roll the value back to the previous value
					if ( prevValue === null ) {
						div.removeAttribute( key );
					} else {
						div.setAttribute( key, prevValue );
					}
					throw store.error();
				}
			}
			return ret;
		});
	}
}() );

// in-memory storage
// fallback for all browsers to enable the API even if we can't persist data
(function() {
	var memory = {};

	function copy( obj ) {
		return obj === undefined ? undefined : JSON.parse( JSON.stringify( obj ) );
	}

	store.addType( "memory", function( key, value, options ) {
		if ( !key ) {
			return copy( memory );
		}

		if ( value === undefined ) {
			return copy( memory[ key ] );
		}

		if ( value === null ) {
			delete memory[ key ];
			return null;
		}

		memory[ key ] = value;
		if ( options.expires ) {
			setTimeout(function() {
				delete memory[ key ];
			}, options.expires );
		}

		return value;
	});
}() );

}( this.amplify = this.amplify || {} ) );

/*!
 * Modernizr v2.0.6
 * http://www.modernizr.com
 *
 * Copyright (c) 2009-2011 Faruk Ates, Paul Irish, Alex Sexton
 * Dual-licensed under the BSD or MIT licenses: www.modernizr.com/license/
 */

/*
 * Modernizr tests which native CSS3 and HTML5 features are available in
 * the current UA and makes the results available to you in two ways:
 * as properties on a global Modernizr object, and as classes on the
 * <html> element. This information allows you to progressively enhance
 * your pages with a granular level of control over the experience.
 *
 * Modernizr has an optional (not included) conditional resource loader
 * called Modernizr.load(), based on Yepnope.js (yepnopejs.com).
 * To get a build that includes Modernizr.load(), as well as choosing
 * which tests to include, go to www.modernizr.com/download/
 *
 * Authors        Faruk Ates, Paul Irish, Alex Sexton, 
 * Contributors   Ryan Seddon, Ben Alman
 */

window.Modernizr = (function( window, document, undefined ) {

    var version = '2.0.6',

    Modernizr = {},
    
    // option for enabling the HTML classes to be added
    enableClasses = true,

    docElement = document.documentElement,
    docHead = document.head || document.getElementsByTagName('head')[0],

    /**
     * Create our "modernizr" element that we do most feature tests on.
     */
    mod = 'modernizr',
    modElem = document.createElement(mod),
    mStyle = modElem.style,

    /**
     * Create the input element for various Web Forms feature tests.
     */
    inputElem = document.createElement('input'),

    smile = ':)',

    toString = Object.prototype.toString,

    // List of property values to set for css tests. See ticket #21
    prefixes = ' -webkit- -moz- -o- -ms- -khtml- '.split(' '),

    // Following spec is to expose vendor-specific style properties as:
    //   elem.style.WebkitBorderRadius
    // and the following would be incorrect:
    //   elem.style.webkitBorderRadius

    // Webkit ghosts their properties in lowercase but Opera & Moz do not.
    // Microsoft foregoes prefixes entirely <= IE8, but appears to
    //   use a lowercase `ms` instead of the correct `Ms` in IE9

    // More here: http://github.com/Modernizr/Modernizr/issues/issue/21
    domPrefixes = 'Webkit Moz O ms Khtml'.split(' '),

    ns = {'svg': 'http://www.w3.org/2000/svg'},

    tests = {},
    inputs = {},
    attrs = {},

    classes = [],

    featureName, // used in testing loop


    // Inject element with style element and some CSS rules
    injectElementWithStyles = function( rule, callback, nodes, testnames ) {

      var style, ret, node,
          div = document.createElement('div');

      if ( parseInt(nodes, 10) ) {
          // In order not to give false positives we create a node for each test
          // This also allows the method to scale for unspecified uses
          while ( nodes-- ) {
              node = document.createElement('div');
              node.id = testnames ? testnames[nodes] : mod + (nodes + 1);
              div.appendChild(node);
          }
      }

      // <style> elements in IE6-9 are considered 'NoScope' elements and therefore will be removed
      // when injected with innerHTML. To get around this you need to prepend the 'NoScope' element
      // with a 'scoped' element, in our case the soft-hyphen entity as it won't mess with our measurements.
      // http://msdn.microsoft.com/en-us/library/ms533897%28VS.85%29.aspx
      style = ['&shy;', '<style>', rule, '</style>'].join('');
      div.id = mod;
      div.innerHTML += style;
      docElement.appendChild(div);

      ret = callback(div, rule);
      div.parentNode.removeChild(div);

      return !!ret;

    },


    // adapted from matchMedia polyfill
    // by Scott Jehl and Paul Irish
    // gist.github.com/786768
    testMediaQuery = function( mq ) {

      if ( window.matchMedia ) {
        return matchMedia(mq).matches;
      }

      var bool;

      injectElementWithStyles('@media ' + mq + ' { #' + mod + ' { position: absolute; } }', function( node ) {
        bool = (window.getComputedStyle ?
                  getComputedStyle(node, null) :
                  node.currentStyle)['position'] == 'absolute';
      });

      return bool;

     },


    /**
      * isEventSupported determines if a given element supports the given event
      * function from http://yura.thinkweb2.com/isEventSupported/
      */
    isEventSupported = (function() {

      var TAGNAMES = {
        'select': 'input', 'change': 'input',
        'submit': 'form', 'reset': 'form',
        'error': 'img', 'load': 'img', 'abort': 'img'
      };

      function isEventSupported( eventName, element ) {

        element = element || document.createElement(TAGNAMES[eventName] || 'div');
        eventName = 'on' + eventName;

        // When using `setAttribute`, IE skips "unload", WebKit skips "unload" and "resize", whereas `in` "catches" those
        var isSupported = eventName in element;

        if ( !isSupported ) {
          // If it has no `setAttribute` (i.e. doesn't implement Node interface), try generic element
          if ( !element.setAttribute ) {
            element = document.createElement('div');
          }
          if ( element.setAttribute && element.removeAttribute ) {
            element.setAttribute(eventName, '');
            isSupported = is(element[eventName], 'function');

            // If property was created, "remove it" (by setting value to `undefined`)
            if ( !is(element[eventName], undefined) ) {
              element[eventName] = undefined;
            }
            element.removeAttribute(eventName);
          }
        }

        element = null;
        return isSupported;
      }
      return isEventSupported;
    })();

    // hasOwnProperty shim by kangax needed for Safari 2.0 support
    var _hasOwnProperty = ({}).hasOwnProperty, hasOwnProperty;
    if ( !is(_hasOwnProperty, undefined) && !is(_hasOwnProperty.call, undefined) ) {
      hasOwnProperty = function (object, property) {
        return _hasOwnProperty.call(object, property);
      };
    }
    else {
      hasOwnProperty = function (object, property) { /* yes, this can give false positives/negatives, but most of the time we don't care about those */
        return ((property in object) && is(object.constructor.prototype[property], undefined));
      };
    }

    /**
     * setCss applies given styles to the Modernizr DOM node.
     */
    function setCss( str ) {
        mStyle.cssText = str;
    }

    /**
     * setCssAll extrapolates all vendor-specific css strings.
     */
    function setCssAll( str1, str2 ) {
        return setCss(prefixes.join(str1 + ';') + ( str2 || '' ));
    }

    /**
     * is returns a boolean for if typeof obj is exactly type.
     */
    function is( obj, type ) {
        return typeof obj === type;
    }

    /**
     * contains returns a boolean for if substr is found within str.
     */
    function contains( str, substr ) {
        return !!~('' + str).indexOf(substr);
    }

    /**
     * testProps is a generic CSS / DOM property test; if a browser supports
     *   a certain property, it won't return undefined for it.
     *   A supported CSS property returns empty string when its not yet set.
     */
    function testProps( props, prefixed ) {
        for ( var i in props ) {
            if ( mStyle[ props[i] ] !== undefined ) {
                return prefixed == 'pfx' ? props[i] : true;
            }
        }
        return false;
    }

    /**
     * testPropsAll tests a list of DOM properties we want to check against.
     *   We specify literally ALL possible (known and/or likely) properties on
     *   the element including the non-vendor prefixed one, for forward-
     *   compatibility.
     */
    function testPropsAll( prop, prefixed ) {

        var ucProp  = prop.charAt(0).toUpperCase() + prop.substr(1),
            props   = (prop + ' ' + domPrefixes.join(ucProp + ' ') + ucProp).split(' ');

        return testProps(props, prefixed);
    }

    /**
     * testBundle tests a list of CSS features that require element and style injection.
     *   By bundling them together we can reduce the need to touch the DOM multiple times.
     */
    /*>>testBundle*/
    var testBundle = (function( styles, tests ) {
        var style = styles.join(''),
            len = tests.length;

        injectElementWithStyles(style, function( node, rule ) {
            var style = document.styleSheets[document.styleSheets.length - 1],
                // IE8 will bork if you create a custom build that excludes both fontface and generatedcontent tests.
                // So we check for cssRules and that there is a rule available
                // More here: https://github.com/Modernizr/Modernizr/issues/288 & https://github.com/Modernizr/Modernizr/issues/293
                cssText = style.cssRules && style.cssRules[0] ? style.cssRules[0].cssText : style.cssText || "",
                children = node.childNodes, hash = {};

            while ( len-- ) {
                hash[children[len].id] = children[len];
            }

            /*>>touch*/           Modernizr['touch'] = ('ontouchstart' in window) || hash['touch'].offsetTop === 9; /*>>touch*/
            /*>>csstransforms3d*/ Modernizr['csstransforms3d'] = hash['csstransforms3d'].offsetLeft === 9;          /*>>csstransforms3d*/
            /*>>generatedcontent*/Modernizr['generatedcontent'] = hash['generatedcontent'].offsetHeight >= 1;       /*>>generatedcontent*/
            /*>>fontface*/        Modernizr['fontface'] = /src/i.test(cssText) &&
                                                                  cssText.indexOf(rule.split(' ')[0]) === 0;        /*>>fontface*/
        }, len, tests);

    })([
        // Pass in styles to be injected into document
        /*>>fontface*/        '@font-face {font-family:"font";src:url("https://")}'         /*>>fontface*/
        
        /*>>touch*/           ,['@media (',prefixes.join('touch-enabled),('),mod,')',
                                '{#touch{top:9px;position:absolute}}'].join('')           /*>>touch*/
                                
        /*>>csstransforms3d*/ ,['@media (',prefixes.join('transform-3d),('),mod,')',
                                '{#csstransforms3d{left:9px;position:absolute}}'].join('')/*>>csstransforms3d*/
                                
        /*>>generatedcontent*/,['#generatedcontent:after{content:"',smile,'";visibility:hidden}'].join('')  /*>>generatedcontent*/
    ],
      [
        /*>>fontface*/        'fontface'          /*>>fontface*/
        /*>>touch*/           ,'touch'            /*>>touch*/
        /*>>csstransforms3d*/ ,'csstransforms3d'  /*>>csstransforms3d*/
        /*>>generatedcontent*/,'generatedcontent' /*>>generatedcontent*/
        
    ]);/*>>testBundle*/


    /**
     * Tests
     * -----
     */

    tests['flexbox'] = function() {
        /**
         * setPrefixedValueCSS sets the property of a specified element
         * adding vendor prefixes to the VALUE of the property.
         * @param {Element} element
         * @param {string} property The property name. This will not be prefixed.
         * @param {string} value The value of the property. This WILL be prefixed.
         * @param {string=} extra Additional CSS to append unmodified to the end of
         * the CSS string.
         */
        function setPrefixedValueCSS( element, property, value, extra ) {
            property += ':';
            element.style.cssText = (property + prefixes.join(value + ';' + property)).slice(0, -property.length) + (extra || '');
        }

        /**
         * setPrefixedPropertyCSS sets the property of a specified element
         * adding vendor prefixes to the NAME of the property.
         * @param {Element} element
         * @param {string} property The property name. This WILL be prefixed.
         * @param {string} value The value of the property. This will not be prefixed.
         * @param {string=} extra Additional CSS to append unmodified to the end of
         * the CSS string.
         */
        function setPrefixedPropertyCSS( element, property, value, extra ) {
            element.style.cssText = prefixes.join(property + ':' + value + ';') + (extra || '');
        }

        var c = document.createElement('div'),
            elem = document.createElement('div');

        setPrefixedValueCSS(c, 'display', 'box', 'width:42px;padding:0;');
        setPrefixedPropertyCSS(elem, 'box-flex', '1', 'width:10px;');

        c.appendChild(elem);
        docElement.appendChild(c);

        var ret = elem.offsetWidth === 42;

        c.removeChild(elem);
        docElement.removeChild(c);

        return ret;
    };

    // On the S60 and BB Storm, getContext exists, but always returns undefined
    // http://github.com/Modernizr/Modernizr/issues/issue/97/

    tests['canvas'] = function() {
        var elem = document.createElement('canvas');
        return !!(elem.getContext && elem.getContext('2d'));
    };

    tests['canvastext'] = function() {
        return !!(Modernizr['canvas'] && is(document.createElement('canvas').getContext('2d').fillText, 'function'));
    };

    // This WebGL test may false positive. 
    // But really it's quite impossible to know whether webgl will succeed until after you create the context. 
    // You might have hardware that can support a 100x100 webgl canvas, but will not support a 1000x1000 webgl 
    // canvas. So this feature inference is weak, but intentionally so.
    
    // It is known to false positive in FF4 with certain hardware and the iPad 2.
    
    tests['webgl'] = function() {
        return !!window.WebGLRenderingContext;
    };

    /*
     * The Modernizr.touch test only indicates if the browser supports
     *    touch events, which does not necessarily reflect a touchscreen
     *    device, as evidenced by tablets running Windows 7 or, alas,
     *    the Palm Pre / WebOS (touch) phones.
     *
     * Additionally, Chrome (desktop) used to lie about its support on this,
     *    but that has since been rectified: http://crbug.com/36415
     *
     * We also test for Firefox 4 Multitouch Support.
     *
     * For more info, see: http://modernizr.github.com/Modernizr/touch.html
     */

    tests['touch'] = function() {
        return Modernizr['touch'];
    };

    /**
     * geolocation tests for the new Geolocation API specification.
     *   This test is a standards compliant-only test; for more complete
     *   testing, including a Google Gears fallback, please see:
     *   http://code.google.com/p/geo-location-javascript/
     * or view a fallback solution using google's geo API:
     *   http://gist.github.com/366184
     */
    tests['geolocation'] = function() {
        return !!navigator.geolocation;
    };

    // Per 1.6:
    // This used to be Modernizr.crosswindowmessaging but the longer
    // name has been deprecated in favor of a shorter and property-matching one.
    // The old API is still available in 1.6, but as of 2.0 will throw a warning,
    // and in the first release thereafter disappear entirely.
    tests['postmessage'] = function() {
      return !!window.postMessage;
    };

    // Web SQL database detection is tricky:

    // In chrome incognito mode, openDatabase is truthy, but using it will
    //   throw an exception: http://crbug.com/42380
    // We can create a dummy database, but there is no way to delete it afterwards.

    // Meanwhile, Safari users can get prompted on any database creation.
    //   If they do, any page with Modernizr will give them a prompt:
    //   http://github.com/Modernizr/Modernizr/issues/closed#issue/113

    // We have chosen to allow the Chrome incognito false positive, so that Modernizr
    //   doesn't litter the web with these test databases. As a developer, you'll have
    //   to account for this gotcha yourself.
    tests['websqldatabase'] = function() {
      var result = !!window.openDatabase;
      /*  if (result){
            try {
              result = !!openDatabase( mod + "testdb", "1.0", mod + "testdb", 2e4);
            } catch(e) {
            }
          }  */
      return result;
    };

    // Vendors had inconsistent prefixing with the experimental Indexed DB:
    // - Webkit's implementation is accessible through webkitIndexedDB
    // - Firefox shipped moz_indexedDB before FF4b9, but since then has been mozIndexedDB
    // For speed, we don't test the legacy (and beta-only) indexedDB
    tests['indexedDB'] = function() {
      for ( var i = -1, len = domPrefixes.length; ++i < len; ){
        if ( window[domPrefixes[i].toLowerCase() + 'IndexedDB'] ){
          return true;
        }
      }
      return !!window.indexedDB;
    };

    // documentMode logic from YUI to filter out IE8 Compat Mode
    //   which false positives.
    tests['hashchange'] = function() {
      return isEventSupported('hashchange', window) && (document.documentMode === undefined || document.documentMode > 7);
    };

    // Per 1.6:
    // This used to be Modernizr.historymanagement but the longer
    // name has been deprecated in favor of a shorter and property-matching one.
    // The old API is still available in 1.6, but as of 2.0 will throw a warning,
    // and in the first release thereafter disappear entirely.
    tests['history'] = function() {
      return !!(window.history && history.pushState);
    };

    tests['draganddrop'] = function() {
        return isEventSupported('dragstart') && isEventSupported('drop');
    };

    // Mozilla is targeting to land MozWebSocket for FF6
    // bugzil.la/659324
    tests['websockets'] = function() {
        for ( var i = -1, len = domPrefixes.length; ++i < len; ){
          if ( window[domPrefixes[i] + 'WebSocket'] ){
            return true;
          }
        }
        return 'WebSocket' in window;
    };


    // http://css-tricks.com/rgba-browser-support/
    tests['rgba'] = function() {
        // Set an rgba() color and check the returned value

        setCss('background-color:rgba(150,255,150,.5)');

        return contains(mStyle.backgroundColor, 'rgba');
    };

    tests['hsla'] = function() {
        // Same as rgba(), in fact, browsers re-map hsla() to rgba() internally,
        //   except IE9 who retains it as hsla

        setCss('background-color:hsla(120,40%,100%,.5)');

        return contains(mStyle.backgroundColor, 'rgba') || contains(mStyle.backgroundColor, 'hsla');
    };

    tests['multiplebgs'] = function() {
        // Setting multiple images AND a color on the background shorthand property
        //  and then querying the style.background property value for the number of
        //  occurrences of "url(" is a reliable method for detecting ACTUAL support for this!

        setCss('background:url(https://),url(https://),red url(https://)');

        // If the UA supports multiple backgrounds, there should be three occurrences
        //   of the string "url(" in the return value for elemStyle.background

        return /(url\s*\(.*?){3}/.test(mStyle.background);
    };


    // In testing support for a given CSS property, it's legit to test:
    //    `elem.style[styleName] !== undefined`
    // If the property is supported it will return an empty string,
    // if unsupported it will return undefined.

    // We'll take advantage of this quick test and skip setting a style
    // on our modernizr element, but instead just testing undefined vs
    // empty string.


    tests['backgroundsize'] = function() {
        return testPropsAll('backgroundSize');
    };

    tests['borderimage'] = function() {
        return testPropsAll('borderImage');
    };


    // Super comprehensive table about all the unique implementations of
    // border-radius: http://muddledramblings.com/table-of-css3-border-radius-compliance

    tests['borderradius'] = function() {
        return testPropsAll('borderRadius');
    };

    // WebOS unfortunately false positives on this test.
    tests['boxshadow'] = function() {
        return testPropsAll('boxShadow');
    };

    // FF3.0 will false positive on this test
    tests['textshadow'] = function() {
        return document.createElement('div').style.textShadow === '';
    };


    tests['opacity'] = function() {
        // Browsers that actually have CSS Opacity implemented have done so
        //  according to spec, which means their return values are within the
        //  range of [0.0,1.0] - including the leading zero.

        setCssAll('opacity:.55');

        // The non-literal . in this regex is intentional:
        //   German Chrome returns this value as 0,55
        // https://github.com/Modernizr/Modernizr/issues/#issue/59/comment/516632
        return /^0.55$/.test(mStyle.opacity);
    };


    tests['cssanimations'] = function() {
        return testPropsAll('animationName');
    };


    tests['csscolumns'] = function() {
        return testPropsAll('columnCount');
    };


    tests['cssgradients'] = function() {
        /**
         * For CSS Gradients syntax, please see:
         * http://webkit.org/blog/175/introducing-css-gradients/
         * https://developer.mozilla.org/en/CSS/-moz-linear-gradient
         * https://developer.mozilla.org/en/CSS/-moz-radial-gradient
         * http://dev.w3.org/csswg/css3-images/#gradients-
         */

        var str1 = 'background-image:',
            str2 = 'gradient(linear,left top,right bottom,from(#9f9),to(white));',
            str3 = 'linear-gradient(left top,#9f9, white);';

        setCss(
            (str1 + prefixes.join(str2 + str1) + prefixes.join(str3 + str1)).slice(0, -str1.length)
        );

        return contains(mStyle.backgroundImage, 'gradient');
    };


    tests['cssreflections'] = function() {
        return testPropsAll('boxReflect');
    };


    tests['csstransforms'] = function() {
        return !!testProps(['transformProperty', 'WebkitTransform', 'MozTransform', 'OTransform', 'msTransform']);
    };


    tests['csstransforms3d'] = function() {

        var ret = !!testProps(['perspectiveProperty', 'WebkitPerspective', 'MozPerspective', 'OPerspective', 'msPerspective']);

        // Webkit’s 3D transforms are passed off to the browser's own graphics renderer.
        //   It works fine in Safari on Leopard and Snow Leopard, but not in Chrome in
        //   some conditions. As a result, Webkit typically recognizes the syntax but
        //   will sometimes throw a false positive, thus we must do a more thorough check:
        if ( ret && 'webkitPerspective' in docElement.style ) {

          // Webkit allows this media query to succeed only if the feature is enabled.
          // `@media (transform-3d),(-o-transform-3d),(-moz-transform-3d),(-ms-transform-3d),(-webkit-transform-3d),(modernizr){ ... }`
          ret = Modernizr['csstransforms3d'];
        }
        return ret;
    };


    tests['csstransitions'] = function() {
        return testPropsAll('transitionProperty');
    };


    /*>>fontface*/
    // @font-face detection routine by Diego Perini
    // http://javascript.nwbox.com/CSSSupport/
    tests['fontface'] = function() {
        return Modernizr['fontface'];
    };
    /*>>fontface*/

    // CSS generated content detection
    tests['generatedcontent'] = function() {
        return Modernizr['generatedcontent'];
    };



    // These tests evaluate support of the video/audio elements, as well as
    // testing what types of content they support.
    //
    // We're using the Boolean constructor here, so that we can extend the value
    // e.g.  Modernizr.video     // true
    //       Modernizr.video.ogg // 'probably'
    //
    // Codec values from : http://github.com/NielsLeenheer/html5test/blob/9106a8/index.html#L845
    //                     thx to NielsLeenheer and zcorpan

    // Note: in FF 3.5.1 and 3.5.0, "no" was a return value instead of empty string.
    //   Modernizr does not normalize for that.

    tests['video'] = function() {
        var elem = document.createElement('video'),
            bool = false;
            
        // IE9 Running on Windows Server SKU can cause an exception to be thrown, bug #224
        try {
            if ( bool = !!elem.canPlayType ) {
                bool      = new Boolean(bool);
                bool.ogg  = elem.canPlayType('video/ogg; codecs="theora"');

                // Workaround required for IE9, which doesn't report video support without audio codec specified.
                //   bug 599718 @ msft connect
                var h264 = 'video/mp4; codecs="avc1.42E01E';
                bool.h264 = elem.canPlayType(h264 + '"') || elem.canPlayType(h264 + ', mp4a.40.2"');

                bool.webm = elem.canPlayType('video/webm; codecs="vp8, vorbis"');
            }
            
        } catch(e) { }
        
        return bool;
    };

    tests['audio'] = function() {
        var elem = document.createElement('audio'),
            bool = false;

        try { 
            if ( bool = !!elem.canPlayType ) {
                bool      = new Boolean(bool);
                bool.ogg  = elem.canPlayType('audio/ogg; codecs="vorbis"');
                bool.mp3  = elem.canPlayType('audio/mpeg;');

                // Mimetypes accepted:
                //   https://developer.mozilla.org/En/Media_formats_supported_by_the_audio_and_video_elements
                //   http://bit.ly/iphoneoscodecs
                bool.wav  = elem.canPlayType('audio/wav; codecs="1"');
                bool.m4a  = elem.canPlayType('audio/x-m4a;') || elem.canPlayType('audio/aac;');
            }
        } catch(e) { }
        
        return bool;
    };


    // Firefox has made these tests rather unfun.

    // In FF4, if disabled, window.localStorage should === null.

    // Normally, we could not test that directly and need to do a
    //   `('localStorage' in window) && ` test first because otherwise Firefox will
    //   throw http://bugzil.la/365772 if cookies are disabled

    // However, in Firefox 4 betas, if dom.storage.enabled == false, just mentioning
    //   the property will throw an exception. http://bugzil.la/599479
    // This looks to be fixed for FF4 Final.

    // Because we are forced to try/catch this, we'll go aggressive.

    // FWIW: IE8 Compat mode supports these features completely:
    //   http://www.quirksmode.org/dom/html5.html
    // But IE8 doesn't support either with local files

    tests['localstorage'] = function() {
        try {
            return !!localStorage.getItem;
        } catch(e) {
            return false;
        }
    };

    tests['sessionstorage'] = function() {
        try {
            return !!sessionStorage.getItem;
        } catch(e){
            return false;
        }
    };


    tests['webworkers'] = function() {
        return !!window.Worker;
    };


    tests['applicationcache'] = function() {
        return !!window.applicationCache;
    };


    // Thanks to Erik Dahlstrom
    tests['svg'] = function() {
        return !!document.createElementNS && !!document.createElementNS(ns.svg, 'svg').createSVGRect;
    };

    // specifically for SVG inline in HTML, not within XHTML
    // test page: paulirish.com/demo/inline-svg
    tests['inlinesvg'] = function() {
      var div = document.createElement('div');
      div.innerHTML = '<svg/>';
      return (div.firstChild && div.firstChild.namespaceURI) == ns.svg;
    };

    // Thanks to F1lt3r and lucideer, ticket #35
    tests['smil'] = function() {
        return !!document.createElementNS && /SVG/.test(toString.call(document.createElementNS(ns.svg, 'animate')));
    };

    tests['svgclippaths'] = function() {
        // Possibly returns a false positive in Safari 3.2?
        return !!document.createElementNS && /SVG/.test(toString.call(document.createElementNS(ns.svg, 'clipPath')));
    };

    // input features and input types go directly onto the ret object, bypassing the tests loop.
    // Hold this guy to execute in a moment.
    function webforms() {
        // Run through HTML5's new input attributes to see if the UA understands any.
        // We're using f which is the <input> element created early on
        // Mike Taylr has created a comprehensive resource for testing these attributes
        //   when applied to all input types:
        //   http://miketaylr.com/code/input-type-attr.html
        // spec: http://www.whatwg.org/specs/web-apps/current-work/multipage/the-input-element.html#input-type-attr-summary
        
        // Only input placeholder is tested while textarea's placeholder is not. 
        // Currently Safari 4 and Opera 11 have support only for the input placeholder
        // Both tests are available in feature-detects/forms-placeholder.js
        Modernizr['input'] = (function( props ) {
            for ( var i = 0, len = props.length; i < len; i++ ) {
                attrs[ props[i] ] = !!(props[i] in inputElem);
            }
            return attrs;
        })('autocomplete autofocus list placeholder max min multiple pattern required step'.split(' '));

        // Run through HTML5's new input types to see if the UA understands any.
        //   This is put behind the tests runloop because it doesn't return a
        //   true/false like all the other tests; instead, it returns an object
        //   containing each input type with its corresponding true/false value

        // Big thanks to @miketaylr for the html5 forms expertise. http://miketaylr.com/
        Modernizr['inputtypes'] = (function(props) {

            for ( var i = 0, bool, inputElemType, defaultView, len = props.length; i < len; i++ ) {

                inputElem.setAttribute('type', inputElemType = props[i]);
                bool = inputElem.type !== 'text';

                // We first check to see if the type we give it sticks..
                // If the type does, we feed it a textual value, which shouldn't be valid.
                // If the value doesn't stick, we know there's input sanitization which infers a custom UI
                if ( bool ) {

                    inputElem.value         = smile;
                    inputElem.style.cssText = 'position:absolute;visibility:hidden;';

                    if ( /^range$/.test(inputElemType) && inputElem.style.WebkitAppearance !== undefined ) {

                      docElement.appendChild(inputElem);
                      defaultView = document.defaultView;

                      // Safari 2-4 allows the smiley as a value, despite making a slider
                      bool =  defaultView.getComputedStyle &&
                              defaultView.getComputedStyle(inputElem, null).WebkitAppearance !== 'textfield' &&
                              // Mobile android web browser has false positive, so must
                              // check the height to see if the widget is actually there.
                              (inputElem.offsetHeight !== 0);

                      docElement.removeChild(inputElem);

                    } else if ( /^(search|tel)$/.test(inputElemType) ){
                      // Spec doesnt define any special parsing or detectable UI
                      //   behaviors so we pass these through as true

                      // Interestingly, opera fails the earlier test, so it doesn't
                      //  even make it here.

                    } else if ( /^(url|email)$/.test(inputElemType) ) {
                      // Real url and email support comes with prebaked validation.
                      bool = inputElem.checkValidity && inputElem.checkValidity() === false;

                    } else if ( /^color$/.test(inputElemType) ) {
                        // chuck into DOM and force reflow for Opera bug in 11.00
                        // github.com/Modernizr/Modernizr/issues#issue/159
                        docElement.appendChild(inputElem);
                        docElement.offsetWidth;
                        bool = inputElem.value != smile;
                        docElement.removeChild(inputElem);

                    } else {
                      // If the upgraded input compontent rejects the :) text, we got a winner
                      bool = inputElem.value != smile;
                    }
                }

                inputs[ props[i] ] = !!bool;
            }
            return inputs;
        })('search tel url email datetime date month week time datetime-local number range color'.split(' '));
    }


    // End of test definitions
    // -----------------------



    // Run through all tests and detect their support in the current UA.
    // todo: hypothetically we could be doing an array of tests and use a basic loop here.
    for ( var feature in tests ) {
        if ( hasOwnProperty(tests, feature) ) {
            // run the test, throw the return value into the Modernizr,
            //   then based on that boolean, define an appropriate className
            //   and push it into an array of classes we'll join later.
            featureName  = feature.toLowerCase();
            Modernizr[featureName] = tests[feature]();

            classes.push((Modernizr[featureName] ? '' : 'no-') + featureName);
        }
    }

    // input tests need to run.
    Modernizr.input || webforms();


    /**
     * addTest allows the user to define their own feature tests
     * the result will be added onto the Modernizr object,
     * as well as an appropriate className set on the html element
     *
     * @param feature - String naming the feature
     * @param test - Function returning true if feature is supported, false if not
     */
     Modernizr.addTest = function ( feature, test ) {
       if ( typeof feature == "object" ) {
         for ( var key in feature ) {
           if ( hasOwnProperty( feature, key ) ) { 
             Modernizr.addTest( key, feature[ key ] );
           }
         }
       } else {

         feature = feature.toLowerCase();

         if ( Modernizr[feature] !== undefined ) {
           // we're going to quit if you're trying to overwrite an existing test
           // if we were to allow it, we'd do this:
           //   var re = new RegExp("\\b(no-)?" + feature + "\\b");  
           //   docElement.className = docElement.className.replace( re, '' );
           // but, no rly, stuff 'em.
           return; 
         }

         test = typeof test == "boolean" ? test : !!test();

         docElement.className += ' ' + (test ? '' : 'no-') + feature;
         Modernizr[feature] = test;

       }

       return Modernizr; // allow chaining.
     };
    

    // Reset modElem.cssText to nothing to reduce memory footprint.
    setCss('');
    modElem = inputElem = null;

    //>>BEGIN IEPP
    // Enable HTML 5 elements for styling (and printing) in IE.
    if ( window.attachEvent && (function(){ var elem = document.createElement('div');
                                            elem.innerHTML = '<elem></elem>';
                                            return elem.childNodes.length !== 1; })() ) {
                                              
        // iepp v2 by @jon_neal & afarkas : github.com/aFarkas/iepp/
        (function(win, doc) {
          win.iepp = win.iepp || {};
          var iepp = win.iepp,
            elems = iepp.html5elements || 'abbr|article|aside|audio|canvas|datalist|details|figcaption|figure|footer|header|hgroup|mark|meter|nav|output|progress|section|summary|time|video',
            elemsArr = elems.split('|'),
            elemsArrLen = elemsArr.length,
            elemRegExp = new RegExp('(^|\\s)('+elems+')', 'gi'),
            tagRegExp = new RegExp('<(\/*)('+elems+')', 'gi'),
            filterReg = /^\s*[\{\}]\s*$/,
            ruleRegExp = new RegExp('(^|[^\\n]*?\\s)('+elems+')([^\\n]*)({[\\n\\w\\W]*?})', 'gi'),
            docFrag = doc.createDocumentFragment(),
            html = doc.documentElement,
            head = html.firstChild,
            bodyElem = doc.createElement('body'),
            styleElem = doc.createElement('style'),
            printMedias = /print|all/,
            body;
          function shim(doc) {
            var a = -1;
            while (++a < elemsArrLen)
              // Use createElement so IE allows HTML5-named elements in a document
              doc.createElement(elemsArr[a]);
          }

          iepp.getCSS = function(styleSheetList, mediaType) {
            if(styleSheetList+'' === undefined){return '';}
            var a = -1,
              len = styleSheetList.length,
              styleSheet,
              cssTextArr = [];
            while (++a < len) {
              styleSheet = styleSheetList[a];
              //currently no test for disabled/alternate stylesheets
              if(styleSheet.disabled){continue;}
              mediaType = styleSheet.media || mediaType;
              // Get css from all non-screen stylesheets and their imports
              if (printMedias.test(mediaType)) cssTextArr.push(iepp.getCSS(styleSheet.imports, mediaType), styleSheet.cssText);
              //reset mediaType to all with every new *not imported* stylesheet
              mediaType = 'all';
            }
            return cssTextArr.join('');
          };

          iepp.parseCSS = function(cssText) {
            var cssTextArr = [],
              rule;
            while ((rule = ruleRegExp.exec(cssText)) != null){
              // Replace all html5 element references with iepp substitute classnames
              cssTextArr.push(( (filterReg.exec(rule[1]) ? '\n' : rule[1]) +rule[2]+rule[3]).replace(elemRegExp, '$1.iepp_$2')+rule[4]);
            }
            return cssTextArr.join('\n');
          };

          iepp.writeHTML = function() {
            var a = -1;
            body = body || doc.body;
            while (++a < elemsArrLen) {
              var nodeList = doc.getElementsByTagName(elemsArr[a]),
                nodeListLen = nodeList.length,
                b = -1;
              while (++b < nodeListLen)
                if (nodeList[b].className.indexOf('iepp_') < 0)
                  // Append iepp substitute classnames to all html5 elements
                  nodeList[b].className += ' iepp_'+elemsArr[a];
            }
            docFrag.appendChild(body);
            html.appendChild(bodyElem);
            // Write iepp substitute print-safe document
            bodyElem.className = body.className;
            bodyElem.id = body.id;
            // Replace HTML5 elements with <font> which is print-safe and shouldn't conflict since it isn't part of html5
            bodyElem.innerHTML = body.innerHTML.replace(tagRegExp, '<$1font');
          };


          iepp._beforePrint = function() {
            // Write iepp custom print CSS
            styleElem.styleSheet.cssText = iepp.parseCSS(iepp.getCSS(doc.styleSheets, 'all'));
            iepp.writeHTML();
          };

          iepp.restoreHTML = function(){
            // Undo everything done in onbeforeprint
            bodyElem.innerHTML = '';
            html.removeChild(bodyElem);
            html.appendChild(body);
          };

          iepp._afterPrint = function(){
            // Undo everything done in onbeforeprint
            iepp.restoreHTML();
            styleElem.styleSheet.cssText = '';
          };



          // Shim the document and iepp fragment
          shim(doc);
          shim(docFrag);

          //
          if(iepp.disablePP){return;}

          // Add iepp custom print style element
          head.insertBefore(styleElem, head.firstChild);
          styleElem.media = 'print';
          styleElem.className = 'iepp-printshim';
          win.attachEvent(
            'onbeforeprint',
            iepp._beforePrint
          );
          win.attachEvent(
            'onafterprint',
            iepp._afterPrint
          );
        })(window, document);
    }
    //>>END IEPP

    // Assign private properties to the return object with prefix
    Modernizr._version      = version;

    // expose these for the plugin API. Look in the source for how to join() them against your input
    Modernizr._prefixes     = prefixes;
    Modernizr._domPrefixes  = domPrefixes;
    
    // Modernizr.mq tests a given media query, live against the current state of the window
    // A few important notes:
    //   * If a browser does not support media queries at all (eg. oldIE) the mq() will always return false
    //   * A max-width or orientation query will be evaluated against the current state, which may change later.
    //   * You must specify values. Eg. If you are testing support for the min-width media query use: 
    //       Modernizr.mq('(min-width:0)')
    // usage:
    // Modernizr.mq('only screen and (max-width:768)')
    Modernizr.mq            = testMediaQuery;   
    
    // Modernizr.hasEvent() detects support for a given event, with an optional element to test on
    // Modernizr.hasEvent('gesturestart', elem)
    Modernizr.hasEvent      = isEventSupported; 

    // Modernizr.testProp() investigates whether a given style property is recognized
    // Note that the property names must be provided in the camelCase variant.
    // Modernizr.testProp('pointerEvents')
    Modernizr.testProp      = function(prop){
        return testProps([prop]);
    };        

    // Modernizr.testAllProps() investigates whether a given style property,
    //   or any of its vendor-prefixed variants, is recognized
    // Note that the property names must be provided in the camelCase variant.
    // Modernizr.testAllProps('boxSizing')    
    Modernizr.testAllProps  = testPropsAll;     


    
    // Modernizr.testStyles() allows you to add custom styles to the document and test an element afterwards
    // Modernizr.testStyles('#modernizr { position:absolute }', function(elem, rule){ ... })
    Modernizr.testStyles    = injectElementWithStyles; 


    // Modernizr.prefixed() returns the prefixed or nonprefixed property name variant of your input
    // Modernizr.prefixed('boxSizing') // 'MozBoxSizing'
    
    // Properties must be passed as dom-style camelcase, rather than `box-sizing` hypentated style.
    // Return values will also be the camelCase variant, if you need to translate that to hypenated style use:
    //
    //     str.replace(/([A-Z])/g, function(str,m1){ return '-' + m1.toLowerCase(); }).replace(/^ms-/,'-ms-');
    
    // If you're trying to ascertain which transition end event to bind to, you might do something like...
    // 
    //     var transEndEventNames = {
    //       'WebkitTransition' : 'webkitTransitionEnd',
    //       'MozTransition'    : 'transitionend',
    //       'OTransition'      : 'oTransitionEnd',
    //       'msTransition'     : 'msTransitionEnd', // maybe?
    //       'transition'       : 'transitionEnd'
    //     },
    //     transEndEventName = transEndEventNames[ Modernizr.prefixed('transition') ];
    
    Modernizr.prefixed      = function(prop){
      return testPropsAll(prop, 'pfx');
    };



    // Remove "no-js" class from <html> element, if it exists:
    docElement.className = docElement.className.replace(/\bno-js\b/, '')
                            
                            // Add the new classes to the <html> element.
                            + (enableClasses ? ' js ' + classes.join(' ') : '');

    return Modernizr;

})(this, this.document);

/**
* Array prototype extensions.
* Extends array prototype with the following methods:
* contains, every, exfiltrate, filter, forEach, getRange, inArray, indexOf, insertAt, map, randomize, removeAt, some, unique
* 
* This extensions doesn't depend on any other code or overwrite existing methods.
* 
*
* Copyright (c) 2007 Harald Hanek (http://js-methods.googlecode.com)
*
* Dual licensed under the MIT (http://www.opensource.org/licenses/mit-license.php)
* and GPL (http://www.gnu.org/licenses/gpl.html) licenses.
* 
* @author Harald Hanek
* @version 0.9
* @lastchangeddate 10. October 2007 15:46:06
* @revision 876
*/

(function(){

	/**
	* Extend the array prototype with the method under the given name if it doesn't currently exist.
	*
	* @private
	*/
	function append(name, method)
	{
		if(!Array.prototype[name])
			Array.prototype[name] = method;
	};


	/**
	* Returns true if every element in 'elements' is in the array.
	*
	* @example [1, 2, 1, 4, 5, 4].contains([1, 2, 4]);
	* @result true
	*
	* @name contains
	* @param Array elements
	* @return Boolean
	*/
	append("contains", function(elements){
		return this.every(function(element){
			return this.indexOf(element) >= 0; }, elements);
	});


	/**
	* Returns the array without the elements in 'elements'.
	* 
	* @example [1, 2, 1, 4, 5, 4].contains([1, 2, 4]);
	* @result true
	*
	* @name exfiltrate
	* @param Array elements
	* @return Boolean
	*/
	append("exfiltrate", function(elements){
		return this.filter(function(element){
			return this.indexOf(element) < 0; }, elements);
	});


	/**
	* Tests whether all elements in the array pass the test implemented by the provided function.
	* 
	* @example [22, 72, 16, 99, 254].every(function(element, index, array) {
	*   return element >= 15;
	* });
	* @result true;
	*
	* @example [12, 72, 16, 99, 254].every(function(element, index, array) {
	*   return element >= 15;
	* });
	* @result false;
	*
	* @name every
	* @param Function fn The function to be called for each element.
	* @param Object scope (optional) The scope of the function (defaults to this).
	* @return Boolean
	*/
	append("every", function(fn, scope){
		for(var i = 0; i < this.length; i++)
			if(!fn.call(scope || window, this[i], i, this))
				return false;
		return true;
	});


	/**
	* Creates a new array with all elements that pass the test implemented by the provided function.
	*
	* Natively supported in Gecko since version 1.8.
	* http://developer.mozilla.org/en/docs/Core_JavaScript_1.5_Reference:Objects:Array:filter
	* 
	* @example [12, 5, 8, 1, 44].filter(function(element, index, array) {
	*   return element >= 10;
	* });
	* @result [12, 44];
	*
	* @name filter
	* @param Function fn The function to be called for each element.
	* @param Object scope (optional) The scope of the function (defaults to this).
	* @return Array
	*/
	append("filter", function(fn, scope){
		var r = [];
		for(var i = 0; i < this.length; i++)
			if(fn.call(scope || window, this[i], i, this))
				r.push(this[i]);
		return r;
	});


	/**
	* Executes a provided function once per array element.
	*
	* Natively supported in Gecko since version 1.8.
	* http://developer.mozilla.org/en/docs/Core_JavaScript_1.5_Reference:Objects:Array:forEach
	* 
	* @example var stuff = "";
	* ["Java", "Script"].forEach(function(element, index, array) {
	*   stuff += element;
	* });
	* @result "JavaScript";
	*
	* @name forEach
	* @param Function fn The function to be called for each element.
	* @param Object scope (optional) The scope of the function (defaults to this).
	* @return void
	*/	
	append("forEach", function(fn, scope){
		for(var i = 0; i < this.length; i++)
			fn.call(scope || window, this[i], i, this);
	});


	/**
	* Returns a range of items in this collection
	*
	* @example [1, 2, 1, 4, 5, 4].getRange(2, 4);
	* @result [1, 4, 5]
	*
	* @name getRange
	* @param Number startIndex (optional) defaults to 0
	* @param Number endIndex (optional) default to the last item
	* @return Array
	*/
	append("getRange", function(start, end){
		var items = this;
		if(items.length < 1)
			return [];

		start = start || 0;
		end = Math.min(typeof end == "undefined" ? this.length-1 : end, this.length-1);
		var r = [];
		if(start <= end)
			for(var i = start; i <= end; i++)
				r[r.length] = items[i];
		else
			for(var i = start; i >= end; i--)
				r[r.length] = items[i];

		return r;
	});


	/**
	* Returns the first index at which a given element can be found in the array, or -1 if it is not present.
	*
	* @example [12, 5, 8, 5, 44].indexOf(5);
	* @result 1;
	*
	* @example [12, 5, 8, 5, 44].indexOf(5, 2);
	* @result 3;
	*
	* @name indexOf
	* @param Object subject Object to search for
	* @param Number offset (optional) Index at which to start searching
	* @return Int
	*/
	append("indexOf", function(subject, offset){
		for(var i = offset || 0; i < this.length; i++)
			if(this[i] === subject)
				return i;
		return -1;
	});


	/**
	* Checks if a given subject can be found in the array.
	*
	* @example [12, 5, 7, 5].inArray(7);
	* @result true;
	*
	* @example [12, 5, 7, 5].inArray(9);
	* @result false;
	*
	* @name inArray
	* @param Object subject Object to search for
	* @return Boolean
	*/
	append("inArray", function(subject){
		for(var i = 0; i < this.length; i++)
			if(subject == this[i])
				return true;
		return false;
	});


	/**
	* Inserts an item at the specified index in the array.
	*
	* @example ['dog', 'cat', 'horse'].insertAt(2, 'mouse');
	* @result ['dog', 'cat', 'mouse', 'horse']
	*
	* @name insertAt
	* @param Number index Position where to insert the element into the array
	* @param Object element The element to insert
	* @return Array
	*/
	append("insertAt", function(index, element){
		for(var k = this.length; k > index; k--)
			this[k] = this[k-1];
		this[index] = element;
		return this;
	});


	/**
	* Creates a new array with the results of calling a provided function on every element in this array.
	*
	* Natively supported in Gecko since version 1.8.
	* http://developer.mozilla.org/en/docs/Core_JavaScript_1.5_Reference:Objects:Array:map
	* 
	* @example ["my", "Name", "is", "HARRY"].map(function(element, index, array) {
	*   return element.toUpperCase();
	* });
	* @result ["MY", "NAME", "IS", "HARRY"];
	*
	* @example [1, 4, 9].map(Math.sqrt);
	* @result [1, 2, 3];
	*
	* @name map
	* @param Function fn The function to be called for each element.
	* @param Object scope (optional) The scope of the function (defaults to this).
	* @return Array
	*/
	append("map", function(fn, scope){
		scope = scope || window;
		var r = [];
		for(var i = 0; i < this.length; i++)
			r[r.length] = fn.call(scope, this[i], i, this);
		return r;
	});


	/**
	* Remove an item from a specified index in the array.
	*
	* @example ['dog', 'cat', 'mouse', 'horse'].deleteAt(2);
	* @result ['dog', 'cat', 'horse']
	*
	* @name removeAt
	* @param Number index The index within the array of the item to remove.
	* @return Array
	*/
	append("removeAt", function(index){
		for(var k = index; k < this.length-1; k++)
			this[k] = this[k+1];
		this.length--;
		return this;
	});


	/**
	* Randomize the order of the elements in the Array.
	*
	* @example [2, 3, 4, 5].randomize();
	* @result [5, 2, 3, 4] randomized result
	*
	* @name randomize
	* @return Array
	*/
	append("randomize", function(){
		return this.sort(function(){return(Math.round(Math.random())-0.5)});
		//return this.sort(function(){return(Math.round(Math.random())-0.5)}, true);
	});


	/**
	* Tests whether some element in the array passes the test implemented by the provided function.
	*
	* Natively supported in Gecko since version 1.8.
	* http://developer.mozilla.org/en/docs/Core_JavaScript_1.5_Reference:Objects:Array:some
	* 
	* @example [101, 199, 250, 200].some(function(element, index, array) {
	*   return element >= 100;
	* });
	* @result true;
	*
	* @example [101, 99, 250, 200].some(function(element, index, array) {
	*   return element >= 100;
	* });
	* @result false;
	*
	* @name some
	* @param Function fn The function to be called for each element.
	* @param Object scope (optional) The scope of the function (defaults to this).
	* @return Boolean
	*/
	append("some", function(fn, scope){
		for(var i = 0; i < this.length; i++)
			if(fn.call(scope || window, this[i], i, this))
				return true;
		return false;
	});


	/**
	* Returns a new array that contains all unique elements of this array.
	*
	* @example [1, 2, 1, 4, 5, 4].unique();
	* @result [1, 2, 4, 5]
	*
	* @name unique
	* @return Array
	*/
	append("unique", function(){
		return this.filter(function(element, index, array){
			return array.indexOf(element) >= index;
		});
	});

})();


/*
Copyright 2011 The greplin-exception-catcher Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
====

This Javascript file lets web applications get stacktraces for all uncaught JS exceptions and send them to Greplin
Exception Catcher.

Features include:
 - Stacktraces in IE 6-8, as well as modern versions of Firefox, Chrome, and Opera
 - Javascript execution entry point information (such as event type and listener) on IE 6-9 and modern versions of
   Firefox, Chrome, Safari, and Opera
 - Redaction of URLs and strings in stacktraces to avoid logging sensitive user information

Things that aren't done yet:
 - Aggregation. Due to the way GEC works now, this would be impossible to do without losing potentially useful
   information. To do this, GEC needs to be able to aggregate based upon a normalized stacktrace while still providing detailed information for each specific incident of the exception.
 - Can't wrap DOM0 events (<div onclick> for example).
 - Some code cleanup: Since this is a small, self-contained project, I took sort of a "hack it until it works" approach
   to coding it. I'd like to go back and structure the code better sometime, but I probably wont' get around to it
   anytime soon since it works very reliably as it is.

How to use it:
1. Create an endpoint at your server to send this stuff to GEC.
2. Modify the call to g.errorCatcher at the end of the file to pass in functions that pass exceptions to GEC and that
   redact URLs respectively. (Note: your URL redaction function will be passed strings that may contain URLs, not bare
   URLs, so keep that in mind)
3. Wrap your JS files if you want to capture errors during their initial execution:
    try {  var your_js_here  }
   catch(e) { window.g && g.handleInitialException && g.handleInitialException(e, '(script filename here)') }
    If you use Closure Compiler, just do
   --output_wrapper="window.COMPILED = true; try { %%output%% } catch(e) { window.g && g.handleInitialException && g.handleInitialException(e, '(script filename here)') }"
4. This exception catching script can't see exceptions that happen before it's loaded, so make sure it's loaded early in
   your page before most of your other scripts.

 */

var g = g || {};


/**
 * Captures uncaught JS exceptions on the page and passes them to GEC.
 * Can capture stacktraces in IE 6-8, Firefox, Chrome, and Opera, and can capture only the top of the stack in IE 9.
 * In Safari, only basic event information is captured.
 * Uses both window.onerror and wrapped DOM prototype interfaces to capture as much information as possible without
 * requiring JS code changes.
 */
g.errorCatcher = function(reportHandler, redactQueryStrings) {
  g.errorCatcher.reportHandler_ = reportHandler;
  g.errorCatcher.redactQueryStrings_ = redactQueryStrings;

  // commented out part is for weird cases where you have two exception catchers.
  // i haven't tested that case at all though, so i'm commenting it out for now.
  var wrappedProperty = 'WrappedListener'; //+ Math.floor(Math.random() * 10000000).toString(30);

  var supportsJsErrorStack;

  try {
    ({})['undefinedMethod']();
  } catch(error) {
    supportsJsErrorStack = 'stack' in error || 'stacktrace' in error;
  }

  var supportsWindowOnerror = 'onerror' in window && !/^Opera/.test(navigator.userAgent);

  var supportsWindowOnerrorStack = /MSIE /.test(navigator.userAgent);

  // Detecting support based on a whitelist sucks, but we don't want to accidentally log personal information, so we
  // only allow browsers that we know that we can redact stacktrace strings for.
  var supportsDOMWrapping =
    // Chrome
    /Chrom(e|ium)/.test(navigator.userAgent) ||

    // IE 9+
    /MSIE (9\.|[1-9][0-9]+\.)/.test(navigator.userAgent) || // XXX compat mode?

    // Firefox 6+
    /Gecko\/[0-9]/.test(navigator.userAgent) && (parseInt(navigator['buildID'], 10) >= 20110830092941) ||

    // Safari 5.1+ (AppleWebKit/534+)
    /AppleWebKit\/(53[4-9]|5[4-9][0-9]|[6-9][0-9]{2}|[1-9][0-9]{3})/.test(navigator.userAgent) ||

    // Opera 11.50+
    /^Opera.*Presto\/(2\.9|[3-9]|[1-9][0-9])/.test(navigator.userAgent);

  if (supportsDOMWrapping) {
    wrapTimeouts();
    wrapDOMEvents();
    wrapXMLHttpRequest();
  }

  if (supportsWindowOnerror &&
      (!supportsDOMWrapping || (!supportsJsErrorStack && supportsWindowOnerrorStack))) {
    window.onerror = function(errorMessage, url, lineNumber) {
      // Grab the error provided by DOM wrappings, if it's available
      var errorObject = g.errorCatcher.lastDomWrapperError_ || {};
      delete g.errorCatcher.lastDomWrapperError_;

      errorObject.message = errorObject.message || errorMessage;
      errorObject.url = errorObject.url || url;
      errorObject.line =  errorObject.line || lineNumber;

      // In IE, get the character offset inside the line of the error from window.event.
      if (window.event && typeof window.event['errorCharacter'] == 'number') {
        errorObject.character = (errorObject.character || window.event['errorCharacter']) + '';
      }

      // If there isn't already a stacktrace generated by the DOM wrappers, try to generate one using the old-fashioned
      // caller method. This only works in IE 6-8. It partially works in IE 9 -- but it only lets you get the top of the
      // stack.
      if (!errorObject.stacktrace && supportsWindowOnerrorStack) {
        try {
          errorObject.stacktrace = g.errorCatcher.getStacktrace(arguments.callee.caller);
        } catch(exception) {
          errorObject.stacktrace = '[error generating stacktrace: ' + exception.message + ']';
        }
      }

      g.errorCatcher.reportException(errorObject);
    };
  }

  /**
   * Wraps setTimeout and setInterval to handle uncaught exceptions in listeners.
   */
  function wrapTimeouts() {
    wrapTimeoutsHelper('setTimeout');
    wrapTimeoutsHelper('setInterval');
    function wrapTimeoutsHelper(timeoutMethodName) {
      var original = window[timeoutMethodName];
      window[timeoutMethodName] = function(listener, delay) {
        if (typeof listener == 'function') {
          var newArgs = Array.prototype.slice.call(arguments);
          newArgs[0] = function() {
            try {
              listener.apply(this, arguments);
            } catch(exception) {
              g.errorCatcher.handleCatchException(
                  exception, timeoutMethodName + '(' + g.errorCatcher.stringify(listener) + ', ' + delay + ')');
            }
          };
          return original.apply(this, newArgs);
        } else {
          // If someone passes a string to setTimeout, don't bother wrapping it.
          return original.apply(this, arguments);
        }
      }
    }
  }


  /**
   * Wraps DOM event interfaces (addEventListener and removeEventListener) to add try/catch wrappers to all event
   * listeners.
   */
  function wrapDOMEvents() {
    var eventsWrappedProperty = 'events' + wrappedProperty;

    wrapDOMEventsHelper(window.XMLHttpRequest.prototype);
    wrapDOMEventsHelper(window.Element.prototype);
    wrapDOMEventsHelper(window);
    wrapDOMEventsHelper(window.document);

    // Workaround for Firefox bug https://bugzilla.mozilla.org/show_bug.cgi?id=456151
    if (document.documentElement.addEventListener != window.Element.prototype.addEventListener) {
      var elementNames =
          ('Unknown,Anchor,Applet,Area,BR,Base,Body,Button,DList,Directory,Div,Embed,FieldSet,Font,Form,Frame,' +
           'FrameSet,HR,Head,Heading,Html,IFrame,Image,Input,IsIndex,LI,Label,Legend,Link,Map,Menu,Meta,Span,OList,' +
           'Object,OptGroup,Option,Paragraph,Param,Pre,Quote,Script,Select,Style,TableCaption,TableCell,TableCol,' +
           'Table,TableRow,TableSection,TextArea,Title,UList,Canvas').split(',');
      elementNames.forEach(function(elementName) {
        var constructor = window['HTML' + elementName + 'Element'];
        if (constructor && constructor.prototype) {
          wrapDOMEventsHelper(constructor.prototype);
        }
      });

    }

    function wrapDOMEventsHelper(object) {
      var originalAddEventListener = object.addEventListener;
      var originalRemoveEventListener = object.removeEventListener;
      if (!originalAddEventListener || !originalRemoveEventListener) {
        return;
      }
      object.addEventListener = function(eventType, listener, useCapture) {
        // Dedupe the listener in case it is already listening unwrapped.
        originalRemoveEventListener.apply(this, arguments);
        if (typeof listener != 'function') {
          // TODO(david): Handle a listener that is not a function, but instead an object that implements the
          // EventListener interface (see http://www.w3.org/TR/DOM-Level-2-Events/events.html#Events-EventListener ).
          originalAddEventListener.apply(this, arguments);
          return;
        }
        listener[eventsWrappedProperty] = listener[eventsWrappedProperty] || {
          innerListener: listener,
          'handleEvent': g.errorCatcher.listenerWrapper_
        };
        originalAddEventListener.call(this, eventType, listener[eventsWrappedProperty], useCapture);
      };
      object.removeEventListener = function(eventType, listener, useCapture) {
        // Remove unwrapped listener, just to be sure.
        originalRemoveEventListener.apply(this, arguments);
        if (typeof listener != 'function') {
          return;
        }
        if (listener[eventsWrappedProperty]) {
          originalRemoveEventListener.call(this, eventType, listener[eventsWrappedProperty], useCapture);
        }
      };
    }
  }


  /**
   * Wrap XMLHttpRequest onreadystatechange listeners to handle uncaught JS exceptions.
   * This only affects the .onreadystatechange property. The addEventListener property is handled by wrapDOMEvents.
   */
  function wrapXMLHttpRequest() {
    var xhrWrappedProperty = 'xhr' + wrappedProperty;
    var ctor = XMLHttpRequest, instance = new XMLHttpRequest;
    if (!/(AppleWebKit|MSIE)/.test(navigator.userAgent) ||
        (Object.getOwnPropertyDescriptor(ctor.prototype, 'onreadystatechange') || {}).configurable &&
         instance.__lookupSetter__ && instance.__lookupSetter__('onreadystatechange')) {
      // The browser has good support for manipulating XMLHttpRequest prototypes.
      var onreadystatechangeSetter = instance.__lookupSetter__('onreadystatechange');
      ctor.prototype.__defineGetter__('onreadystatechange', function() {
        return this[xhrWrappedProperty];
      });
      ctor.prototype.__defineSetter__('onreadystatechange', function(listener) {
        this[xhrWrappedProperty] = listener;
        onreadystatechangeSetter.call(this, wrappedReadyStateChange);
      });
    } else {
      // Chrome and Safari have problems with this. Instead, check to see if onreadystatechange needs to be wrapped
      // from a readystatechange event listener.
      var send = instance.send;
      var addEventListener = instance.addEventListener;
      XMLHttpRequest.prototype.send = function() {
        addEventListener.call(this, 'readystatechange', wrapReadyStateChange, true);
        return send.apply(this, arguments);
      }
    }
    function wrappedReadyStateChange() {
      try {
        var onreadystatechange =
            (this.onreadystatechange == arguments.callee ?
             this[xhrWrappedProperty] : this.onreadystatechange);
        this[xhrWrappedProperty].apply(this, arguments);
      } catch(exception) {
        // TODO(david): Expose some information about the xmlhttprequest to the exception logging (maybe request url)
        g.errorCatcher.handleCatchException(exception, 'onreadystatechange');
      }
    }
    // Used in the wrapped XHR::send handler to wrap onreadystatechange in response to addEventListener
    // readystatechange events that fire first.
    function wrapReadyStateChange() {
      if (this.onreadystatechange && this.onreadystatechange != wrappedReadyStateChange) {
        this[xhrWrappedProperty] = this.onreadystatechange;
        this.onreadystatechange = wrappedReadyStateChange;
      }
    }
  }
};


/**
 * Time that the last error was reported. Used for rate-limiting.
 * @type {number}
 */
g.errorCatcher.lastError_ = 0;


/**
 * Delay between reporting errors. Increases dynamically.
 * @type {number}
 */
g.errorCatcher.errorDelay_ = 10;


/**
 * Wrapper for addEventListener/removeEventListener listeners. Global to avoid potential memory/performance impacts of a
 * function closure for each event listener. This is a handleEvent property of the EventHandler object passed to
 * addEventListener. It accesses other properties of that object to read exception information.
 * @param {Event} eventObject The DOM event.
 */
g.errorCatcher.listenerWrapper_ = function(eventObject) {
  try {
    return this.innerListener.apply(eventObject.target, arguments);
  } catch(exception) {
    g.errorCatcher.handleCatchException(
        exception, eventObject.type + ' listener ' + g.errorCatcher.stringify(this.innerListener) + ' on ' +
            g.errorCatcher.stringify(eventObject.currentTarget));
  }
};


/**
 * Passes an exception to GEC.
 * TODO(david): show a message to the user. Let the user elect to send more detailed error information (un-redacted
 * strings).
 * @param {Object} errorObject An object describing the error.
 */
g.errorCatcher.reportException = function(errorObject) {
  var d = (new Date).getTime();
  if (d - g.errorCatcher.lastError_ < g.errorCatcher.errorDelay_) {
    // Rate limited
    return;
  }
  g.errorCatcher.lastError_ = d;
  g.errorCatcher.errorDelay_ = g.errorCatcher.errorDelay_ * 2;
  errorObj = {
      'msg':g.errorCatcher.redactQueryStrings_(errorObject.message || ''),
      'line': errorObject.line + (typeof errorObject.character == 'string' ? ':' + errorObject.character : ''),
      'trace':'Type: ' + errorObject.name + '\nUser-agent: ' + navigator.userAgent +
          '\nURL: ' + g.errorCatcher.redactQueryStrings_(location.href) + '\n\n' +
          g.errorCatcher.redactQueryStrings_(errorObject.stacktrace || ''),
      'ts': Math.floor(new Date().getTime() / 1000),
      'name':g.errorCatcher.redactQueryStrings_(errorObject.context || '') || 'unidentified JS thread'};
  g.errorCatcher.reportHandler_(errorObj);

};


/**
 * Handles exceptions from the try { } catch { } block added around all of our compiled JS by our Closure Compiler
 * configuration. This handles exceptions that occur during the intiial execution of the script.
 * @param {Error} caughtException The caught exception.
 * @param {string} fileName The name of the JS file where the exception occured.
 */
g.errorCatcher.handleInitialException = function(caughtException, fileName) {
  g.errorCatcher.handleCatchException(caughtException, 'Initial execution of ' + fileName);
};


/**
 * Handles a caught exception. When window.onerror is available, the exception is re-thrown so that additional
 * information from window.onerror can be added. Otherwise, the exception is passed to reportException, where it is
 * sent to GEC and potentially displayed to the user.
 * @param {Error} caughtException The caught JS exception.
 * @param context
 */
g.errorCatcher.handleCatchException = function(caughtException, context) {
  if (!(caughtException instanceof window.Error)) {
    caughtException = new Error(caughtException);
  }

  var errorObject = {};
  errorObject.context = context;
  errorObject.name = caughtException.name;
  // Opera has both stacktrace and stack. Stacktrace is much more detailed, so use that when available.
  errorObject.stacktrace = caughtException['stacktrace'] || caughtException['stack'];
  if (/Gecko/.test(navigator.userAgent) && !/AppleWebKit/.test(navigator.userAgent)) {
    errorObject.stacktrace = g.errorCatcher.redactFirefoxStacktraceStrings(errorObject.stacktrace);
  }
  errorObject.message = caughtException.message;
  errorObject.number = caughtException.number;

  var matches;
  if ('lineNumber' in caughtException) {
    errorObject.line = caughtException['lineNumber'];
  } else if ('line' in caughtException) {
    errorObject.line = caughtException['line'];
  } else if (/Chrom(e|ium)/.test(navigator.userAgent)) {
    matches = caughtException.stack.match(/\:(\d+)\:(\d+)\)(\n|$)/);
    if (matches) {
      errorObject.line = matches[1];
      errorObject.character = matches[2];
    }
  } else if (/Opera/.test(navigator.userAgent)) {
    matches = (errorObject['stacktrace'] || '').match(/Error thrown at line (\d+), column (\d+)/);
    if (matches) {
      errorObject.line = matches[1];
      errorObject.character = matches[2];
    } else {
      matches = (errorObject['stacktrace'] || '').match(/Error thrown at line (\d+)/);
      if (matches){
        errorObject.line = matches[1];
      }
    }
  }

  if (window.onerror) {
    // window.onerror is still needed to get stack in IE, so we need to re-throw the error to that.
    g.errorCatcher.lastDomWrapperError_ = errorObject;
    throw caughtException;
  } else {
    g.errorCatcher.reportException(errorObject);
  }
};


/**
 * @param {Function} opt_topFunction The function at the top of the stack; if omitted, the caller of makeStacktrace is
 *     used.
 * @return {string} A string showing the stack of functions and arguments.
 */
g.errorCatcher.getStacktrace = function(opt_topFunction) {
  var stacktrace = '';
  var func = opt_topFunction || arguments.callee.caller;
  var used = [];
  var length = 0;
  stacktraceLoop: do {
    stacktrace += g.errorCatcher.getFunctionName(func) + g.errorCatcher.getFunctionArgumentsString(func) + '\n';
    used.push(func);
    try {
      func = func.caller;
      for (var i = 0; i < used.length; i++) {
        if (used[i] == func) {
          stacktrace += g.errorCatcher.getFunctionName(func) + '(???)\n(...)\n';
          break stacktraceLoop;
        }
      }
    } catch(exception) {
      stacktrace += '(???' + exception.message + ')\n';
      break stacktraceLoop;
    }
    if (length > 50) {
      stacktrace += '(...)\n';
    }
  } while (func);
  return stacktrace;
};


/**
 * @param {string} string The string to shorten.
 * @param {number} maxLength The maximum length of the new string.
 * @return {string} The string, shortened if it exceeds maxLength.
 */
g.errorCatcher.shortenString = function(string, maxLength) {
  if (string.length > maxLength) {
    string = string.substr(0, maxLength) + '...';
  }
  return string;
};


/**
 * @param {Function} func The function to get the name of.
 * @return {string} The name of the function, or a snippet of the function's source code if it is an anonymous function.
 */
g.errorCatcher.getFunctionName = function(func) {
  var name;
  try {
    if ('name' in Function.prototype && func.name) {
      name = func.name;
    } else {
      var funcStr = func.toString();
      var matches = /function ([^\(]+)/.exec(funcStr);
      name = matches && matches[1] || '[anonymous function: ' + g.errorCatcher.shortenString(func.toString(), 90) + ']';
    }
  } catch(exception) {
    name = '[inaccessible function]'
  }
  return name;
};


/**
 * @param func The function to get a string describing the arguments for. Must be in the current callstack.
 * @return {string} A string of the arguments passed to the function.
 */
g.errorCatcher.getFunctionArgumentsString = function(func) {
  var argsStrings = [];
  try {
    var args = func.arguments;
    if (args) {
      for (var i = 0, length = args.length; i < length; i++) {
        argsStrings.push(g.errorCatcher.stringify(args[i]));
      }
    }
  } catch(exception) {
    argsStrings.push('...?');
  }
  return '(' + argsStrings.join(',') + ')';
};


/**
 * Converts objects and primitives to strings describing them. String inputs are redacted.
 * @param {*} thing The object or primitive to describe.
 * @return {string} String describing the input.
 */
g.errorCatcher.stringify = function(thing) {
  var string = '[???]';
  try {
    var type = typeof thing;
    string = '[' + type + '?]';
    switch (type) {
      case 'undefined':
          string = 'undefined';
          break;
      case 'number':
      case 'boolean':
          string = thing.toString();
          break;
      case 'object':
          if (thing == null) {
            string = 'null';
            break;
          }
          if (thing instanceof Date) {
            string = 'new Date("' + thing.toString() + '")';
            break;
          }
          var toStringValue = thing.toString();
          if (/^\[[a-z ]*\]$/i.test(toStringValue)) {
            string = toStringValue;
            break;
          }
          if (typeof thing.length == 'number') {
            string = '[arraylike object, length = ' + thing.length + ']';
            break;
          }
          string = '[object]';
          break;
      case 'string':
          string = '"' + g.errorCatcher.redactString(thing) + '"';
          break;
      case 'function':
          string = '/* function */ ' + g.errorCatcher.getFunctionName(thing);
          break;
      default:
          string = '[' + type + '???]';
          break;
    }
  } catch(exception) { }
  return string;
};


/**
 * Finds quoted strings in a Firefox stacktrace and replaces them with redacted versions. Handles pesky escaped quotes
 * too. This relies on Firefox's specific stringification/escaping behavior and might not work as consistently in other
 * browsers.
 * @param {string} stacktraceStr The stacktrace to redact strings from.
 * @return {string} The stacktrace, with strings redacted.
 */
g.errorCatcher.redactFirefoxStacktraceStrings = function(stacktraceStr) {
  if (!/\"/.test(stacktraceStr)) {
    return stacktraceStr;
  }
  // We can safely use new ecmascript array methods because this code only runs in Firefox.
  return stacktraceStr.split('\n').map(function(stacktraceLine) {
    var quoteLocations = [];
    var index = 0;
    do {
      index = (stacktraceLine.indexOf('"', index + 1));
      if (index != -1) {
        quoteLocations.push(index);
      }
    } while (index != -1);
    quoteLocations = quoteLocations.filter(function(quoteLocation) {
      var backslashCount = 0, index = quoteLocation;
      while (index--) {
        if (stacktraceLine.charAt(index) != '\\') {
          break;
        }
        backslashCount = backslashCount + 1;
      }
      // If a quotation mark is preceded by a non-even number of backslashes, it is escaped. Otherwise, only the
      // backslashes are escaped.
      // \"    escaped quote
      // \\"   escaped backslash, unescaped quote
      // \\\"  escaped backslash, escaped quote
      // (etc)
      return (backslashCount % 2 == 0);
    });
    if (quoteLocations.length % 2 == 1) {
      quoteLocations.push(stacktraceLine.length);
    }
    for (var i = quoteLocations.length - 1; i > 0; i -= 2) {
      stacktraceLine = stacktraceLine.substr(0, quoteLocations[i - 1] + 1) +
          g.errorCatcher.redactString(stacktraceLine.substring(quoteLocations[i - 1] + 1, quoteLocations[i])) +
          stacktraceLine.substr(quoteLocations[i]);
    }
    return stacktraceLine;
  }).join('\n');
};


/**
 * Redacts a string for user privacy.
 * @param {string} str The string to redact.
 * @return {string} The redacted string.
 */
g.errorCatcher.redactString = function(str) {
  return '[string redacted]';
  // This commented out alternative attempts to at least make certain types of string (HTML, for example) maintain a
  // recognizable pattern.
  // return g.errorCatcher.shortenString(str.replace(/[a-z]/g, 'x').replace(/[A-Z]/g, 'X').replace(/[0-9]/g, '#').replace(
  //    /[^\\\s\[\]<>xX\"\'\(\)\.\,\?\!\#\=\:\;\&\|\@\_\-]/g, '*'), 150).replace(/\r/g, '').replace(/\n/g, '\\n');
};

// g.errorCatcher can cause problems with debuggers (it breaks the Firebug console, for example), so it should be
// disabled in development environments. This if statements g.errorCatcher if you're using
if (!/dev/.test(window.location.host)) {
  g.errorCatcher(function(errorObj) {
    var key = '27461631-f992-4f72-b94d-b98996ef1a53';
    var host = 'https://logs.loggly.com';
    castor = new loggly({url: host+'/inputs/'+key+'?rt=1', level: 'log'});
    castor.error(JSON.stringify({host: window.location.host, error: errorObj}));
  }, function(str) {
    // this is the URL redaction function. this one just removes ?q= paramter values, but you should adapt this to your own application if needed.
    return str.replace(/([\#\?\&][Qq]\=)[^\=\&\#\s]*/g, '$1[redacted]');
  });
}

/*
 * Copyright 2010 Matthew Eernisse (mde@fleegix.org)
 * and Open Source Applications Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Credits: Ideas included from incomplete JS implementation of Olson
 * parser, "XMLDAte" by Philippe Goetz (philippe.goetz@wanadoo.fr)
 *
 * Contributions:
 * Jan Niehusmann
 * Ricky Romero
 * Preston Hunt (prestonhunt@gmail.com),
 * Dov. B Katz (dov.katz@morganstanley.com),
 * Peter Bergström (pbergstr@mac.com)
*/

if (typeof timezoneJS == 'undefined') { timezoneJS = {}; }

timezoneJS.Date = function () {
  var args = Array.prototype.slice.apply(arguments);
  var t = null;
  var dt = null;
  var tz = null;
  var utc = false;

  // No args -- create a floating date based on the current local offset
  if (args.length === 0) {
    dt = new Date();
  }
  // Date string or timestamp -- assumes floating
  else if (args.length == 1) {
    dt = new Date(args[0]);
  }
  // year, month, [date,] [hours,] [minutes,] [seconds,] [milliseconds,] [tzId,] [utc]
  else {
    t = args[args.length-1];
    // Last arg is utc
    if (typeof t == 'boolean') {
      utc = args.pop();
      tz = args.pop();
    }
    // Last arg is tzId
    else if (typeof t == 'string') {
      tz = args.pop();
      if (tz == 'Etc/UTC' || tz == 'Etc/GMT') {
        utc = true;
      }
    }

    // Date string (e.g., '12/27/2006')
    t = args[args.length-1];
    if (typeof t == 'string') {
      dt = new Date(args[0]);
    }
    // Date part numbers
    else {
      var a = [];
      for (var i = 0; i < 8; i++) {
        a[i] = args[i] || 0;
      }
      dt = new Date(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    }
  }
  this._useCache = false;
  this._tzInfo = {};
  this._tzAbbr = '';
  this._day = 0;
  this.year = 0;
  this.month = 0;
  this.date = 0;
  this.hours= 0;
  this.minutes = 0;
  this.seconds = 0;
  this.milliseconds = 0;
  this.timezone = tz || null;
  this.utc = utc || false;
  this.setFromDateObjProxy(dt);
};

timezoneJS.Date.prototype = {
  getDate: function () { return this.date; },
  getDay: function () { return this._day; },
  getFullYear: function () { return this.year; },
  getMonth: function () { return this.month; },
  getYear: function () { return this.year; },
  getHours: function () {
    return this.hours;
  },
  getMilliseconds: function () {
    return this.milliseconds;
  },
  getMinutes: function () {
    return this.minutes;
  },
  getSeconds: function () {
    return this.seconds;
  },
  getTime: function () {
    var dt = Date.UTC(this.year, this.month, this.date,
      this.hours, this.minutes, this.seconds, this.milliseconds);
    return dt + (this.getTimezoneOffset()*60*1000);
  },
  getTimezone: function () {
    return this.timezone;
  },
  getTimezoneOffset: function () {
    var info = this.getTimezoneInfo();
    return info.tzOffset;
  },
  getTimezoneAbbreviation: function () {
    var info = this.getTimezoneInfo();
    return info.tzAbbr;
  },
  getTimezoneInfo: function () {
    var res;
    if (this.utc) {
      res = { tzOffset: 0,
        tzAbbr: 'UTC' };
    }
    else {
      if (this._useCache) {
        res = this._tzInfo;
      }
      else {
        if (this.timezone) {
          var dt = new Date(Date.UTC(this.year, this.month, this.date,
            this.hours, this.minutes, this.seconds, this.milliseconds));
          var tz = this.timezone;
          res = timezoneJS.timezone.getTzInfo(dt, tz);
        }
        // Floating -- use local offset
        else {
          res = { tzOffset: this.getLocalOffset(),
            tzAbbr: null };
        }
        this._tzInfo = res;
        this._useCache = true;
      }
    }
    return res;
  },
  getUTCDate: function () {
    return this.getUTCDateProxy().getUTCDate();
  },
  getUTCDay: function () {
    return this.getUTCDateProxy().getUTCDay();
  },
  getUTCFullYear: function () {
    return this.getUTCDateProxy().getUTCFullYear();
  },
  getUTCHours: function () {
    return this.getUTCDateProxy().getUTCHours();
  },
  getUTCMilliseconds: function () {
    return this.getUTCDateProxy().getUTCMilliseconds();
  },
  getUTCMinutes: function () {
    return this.getUTCDateProxy().getUTCMinutes();
  },
  getUTCMonth: function () {
    return this.getUTCDateProxy().getUTCMonth();
  },
  getUTCSeconds: function () {
    return this.getUTCDateProxy().getUTCSeconds();
  },
  setDate: function (n) {
    this.setAttribute('date', n);
  },
  setFullYear: function (n) {
    this.setAttribute('year', n);
  },
  setMonth: function (n) {
    this.setAttribute('month', n);
  },
  setYear: function (n) {
    this.setUTCAttribute('year', n);
  },
  setHours: function (n) {
    this.setAttribute('hours', n);
  },
  setMilliseconds: function (n) {
    this.setAttribute('milliseconds', n);
  },
  setMinutes: function (n) {
    this.setAttribute('minutes', n);
  },
  setSeconds: function (n) {
    this.setAttribute('seconds', n);
  },
  setTime: function (n) {
    if (isNaN(n)) { throw new Error('Units must be a number.'); }
    var dt = new Date(0);
    dt.setUTCMilliseconds(n - (this.getTimezoneOffset()*60*1000));
    this.setFromDateObjProxy(dt, true);
  },
  setUTCDate: function (n) {
    this.setUTCAttribute('date', n);
  },
  setUTCFullYear: function (n) {
    this.setUTCAttribute('year', n);
  },
  setUTCHours: function (n) {
    this.setUTCAttribute('hours', n);
  },
  setUTCMilliseconds: function (n) {
    this.setUTCAttribute('milliseconds', n);
  },
  setUTCMinutes: function (n) {
    this.setUTCAttribute('minutes', n);
  },
  setUTCMonth: function (n) {
    this.setUTCAttribute('month', n);
  },
  setUTCSeconds: function (n) {
    this.setUTCAttribute('seconds', n);
  },
  toGMTString: function () {},
  toLocaleString: function () {},
  toLocaleDateString: function () {},
  toLocaleTimeString: function () {},
  toSource: function () {},
  toString: function () {
    // Get a quick looky at what's in there
    var str = this.getFullYear() + '-' + (this.getMonth()+1) + '-' + this.getDate();
    var hou = this.getHours() || 12;
    hou = String(hou);
    var min = String(this.getMinutes());
    if (min.length == 1) { min = '0' + min; }
    var sec = String(this.getSeconds());
    if (sec.length == 1) { sec = '0' + sec; }
    str += ' ' + hou;
    str += ':' + min;
    str += ':' + sec;
    return str;
  },
  toUTCString: function () {},
  valueOf: function () {
    return this.getTime();
  },
  clone: function () {
    return new timezoneJS.Date(this.year, this.month, this.date,
      this.hours, this.minutes, this.seconds, this.milliseconds,
      this.timezone);
  },
  setFromDateObjProxy: function (dt, fromUTC) {
    this.year = fromUTC ? dt.getUTCFullYear() : dt.getFullYear();
    this.month = fromUTC ? dt.getUTCMonth() : dt.getMonth();
    this.date = fromUTC ? dt.getUTCDate() : dt.getDate();
    this.hours = fromUTC ? dt.getUTCHours() : dt.getHours();
    this.minutes = fromUTC ? dt.getUTCMinutes() : dt.getMinutes();
    this.seconds = fromUTC ? dt.getUTCSeconds() : dt.getSeconds();
    this.milliseconds = fromUTC ? dt.getUTCMilliseconds() : dt.getMilliseconds();
    this._day = fromUTC ? dt.getUTCDay() : dt.getDay();
    this._useCache = false;
  },
  getUTCDateProxy: function () {
    var dt = new Date(Date.UTC(this.year, this.month, this.date,
      this.hours, this.minutes, this.seconds, this.milliseconds));
    dt.setUTCMinutes(dt.getUTCMinutes() + this.getTimezoneOffset());
    return dt;
  },
  setAttribute: function (unit, n) {
    if (isNaN(n)) { throw new Error('Units must be a number.'); }
    var dt = new Date(this.year, this.month, this.date,
      this.hours, this.minutes, this.seconds, this.milliseconds);
    var meth = unit == 'year' ? 'FullYear' : unit.substr(0, 1).toUpperCase() +
      unit.substr(1);
    dt['set' + meth](n);
    this.setFromDateObjProxy(dt);
  },
  setUTCAttribute: function (unit, n) {
    if (isNaN(n)) { throw new Error('Units must be a number.'); }
    var meth = unit == 'year' ? 'FullYear' : unit.substr(0, 1).toUpperCase() +
      unit.substr(1);
    var dt = this.getUTCDateProxy();
    dt['setUTC' + meth](n);
    dt.setUTCMinutes(dt.getUTCMinutes() - this.getTimezoneOffset());
    this.setFromDateObjProxy(dt, true);
  },
  setTimezone: function (tz) {
    if (tz == 'Etc/UTC' || tz == 'Etc/GMT') {
      this.utc = true;
    } else {
      this.utc = false;
    }
    this.timezone = tz;
    this._useCache = false;
  },
  removeTimezone: function () {
    this.utc = false;
    this.timezone = null;
    this._useCache = false;
  },
  civilToJulianDayNumber: function (y, m, d) {
    var a;
    // Adjust for zero-based JS-style array
    m++;
    if (m > 12) {
      a = parseInt(m/12, 10);
      m = m % 12;
      y += a;
    }
    if (m <= 2) {
      y -= 1;
      m += 12;
    }
    a = Math.floor(y / 100);
    var b = 2 - a + Math.floor(a / 4);
    jDt = Math.floor(365.25 * (y + 4716)) +
      Math.floor(30.6001 * (m + 1)) +
      d + b - 1524;
    return jDt;
  },
  getLocalOffset: function () {
    var dt = this;
    var d = new Date(dt.getYear(), dt.getMonth(), dt.getDate(),
      dt.getHours(), dt.getMinutes(), dt.getSeconds());
    return d.getTimezoneOffset();
  },
  convertToTimezone: function(tz) {
    var dt = new Date();
    res = timezoneJS.timezone.getTzInfo(dt, tz);
    
    convert_offset = this.getTimezoneOffset() - res.tzOffset // offset in minutes
    converted_date = new timezoneJS.Date(this + convert_offset*60*1000)
    this.setFromDateObjProxy(converted_date, true)
    this.setTimezone(tz)
  }
};

timezoneJS.timezone = new function() {
  var _this = this;
  var monthMap = { 'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3,'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11 };
  var dayMap = {'sun': 0,'mon' :1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6 };
  var regionMap = {'EST':'northamerica','MST':'northamerica','HST':'northamerica','EST5EDT':'northamerica','CST6CDT':'northamerica','MST7MDT':'northamerica','PST8PDT':'northamerica','America':'northamerica','Pacific':'australasia','Atlantic':'europe','Africa':'africa','Indian':'africa','Antarctica':'antarctica','Asia':'asia','Australia':'australasia','Europe':'europe','WET':'europe','CET':'europe','MET':'europe','EET':'europe'};
  var regionExceptions = {'Pacific/Honolulu':'northamerica','Atlantic/Bermuda':'northamerica','Atlantic/Cape_Verde':'africa','Atlantic/St_Helena':'africa','Indian/Kerguelen':'antarctica','Indian/Chagos':'asia','Indian/Maldives':'asia','Indian/Christmas':'australasia','Indian/Cocos':'australasia','America/Danmarkshavn':'europe','America/Scoresbysund':'europe','America/Godthab':'europe','America/Thule':'europe','Asia/Yekaterinburg':'europe','Asia/Omsk':'europe','Asia/Novosibirsk':'europe','Asia/Krasnoyarsk':'europe','Asia/Irkutsk':'europe','Asia/Yakutsk':'europe','Asia/Vladivostok':'europe','Asia/Sakhalin':'europe','Asia/Magadan':'europe','Asia/Kamchatka':'europe','Asia/Anadyr':'europe','Africa/Ceuta':'europe','America/Argentina/Buenos_Aires':'southamerica','America/Argentina/Cordoba':'southamerica','America/Argentina/Tucuman':'southamerica','America/Argentina/La_Rioja':'southamerica','America/Argentina/San_Juan':'southamerica','America/Argentina/Jujuy':'southamerica','America/Argentina/Catamarca':'southamerica','America/Argentina/Mendoza':'southamerica','America/Argentina/Rio_Gallegos':'southamerica','America/Argentina/Ushuaia':'southamerica','America/Aruba':'southamerica','America/La_Paz':'southamerica','America/Noronha':'southamerica','America/Belem':'southamerica','America/Fortaleza':'southamerica','America/Recife':'southamerica','America/Araguaina':'southamerica','America/Maceio':'southamerica','America/Bahia':'southamerica','America/Sao_Paulo':'southamerica','America/Campo_Grande':'southamerica','America/Cuiaba':'southamerica','America/Porto_Velho':'southamerica','America/Boa_Vista':'southamerica','America/Manaus':'southamerica','America/Eirunepe':'southamerica','America/Rio_Branco':'southamerica','America/Santiago':'southamerica','Pacific/Easter':'southamerica','America/Bogota':'southamerica','America/Curacao':'southamerica','America/Guayaquil':'southamerica','Pacific/Galapagos':'southamerica','Atlantic/Stanley':'southamerica','America/Cayenne':'southamerica','America/Guyana':'southamerica','America/Asuncion':'southamerica','America/Lima':'southamerica','Atlantic/South_Georgia':'southamerica','America/Paramaribo':'southamerica','America/Port_of_Spain':'southamerica','America/Montevideo':'southamerica','America/Caracas':'southamerica'};

  function invalidTZError(t) {
    throw new Error('Timezone "' + t + '" is either incorrect, or not loaded in the timezone registry.');
  }
  function getRegionForTimezone(tz) {
    var exc = regionExceptions[tz];
    var ret;
    if (exc) {
      return exc;
    }
    else {
      reg = tz.split('/')[0];
      ret = regionMap[reg];
      // If there's nothing listed in the main regions for
      // this TZ, check the 'backward' links
      if (!ret) {
        var link = _this.zones[tz];
        if (typeof link == 'string') {
          return getRegionForTimezone(link);
        }
      }
      return ret;
    }
  }
  function parseTimeString(str) {
    var pat = /(\d+)(?::0*(\d*))?(?::0*(\d*))?([wsugz])?$/;
    var hms = str.match(pat);
    hms[1] = parseInt(hms[1], 10);
    hms[2] = hms[2] ? parseInt(hms[2], 10) : 0;
    hms[3] = hms[3] ? parseInt(hms[3], 10) : 0;
    return hms;
  }
  function getZone(dt, tz) {
    var t = tz;
    var zoneList = _this.zones[t];
    // Follow links to get to an acutal zone
    while (typeof zoneList == "string") {
      t = zoneList;
      zoneList = _this.zones[t];
    }
    for(var i = 0; i < zoneList.length; i++) {
      var z = zoneList[i];
      if (!z[3]) { break; }
      var yea = parseInt(z[3], 10);
      var mon = 11;
      var dat = 31;
      if (z[4]) {
        mon = monthMap[z[4].substr(0, 3).toLowerCase()];
        dat = parseInt(z[5], 10);
      }
      var t = z[6] ? z[6] : '23:59:59';
      t = parseTimeString(t);
      var d = Date.UTC(yea, mon, dat, t[1], t[2], t[3]);
      if (dt.getTime() < d) { break; }
    }
    if (i == zoneList.length) { throw new Error('No Zone found for "' + timezone + '" on ' + dt); }
    return zoneList[i];

  }
  function getBasicOffset(z) {
    var off = parseTimeString(z[0]);
    var adj = z[0].indexOf('-') == 0 ? -1 : 1
    off = adj * (((off[1] * 60 + off[2]) *60 + off[3]) * 1000);
    return -off/60/1000;
  }

  // if isUTC is true, date is given in UTC, otherwise it's given
  // in local time (ie. date.getUTC*() returns local time components)
  function getRule( date, zone, isUTC ) {
    var ruleset = zone[1];
    var basicOffset = getBasicOffset( zone );

    // Convert a date to UTC. Depending on the 'type' parameter, the date
    // parameter may be:
    // 'u', 'g', 'z': already UTC (no adjustment)
    // 's': standard time (adjust for time zone offset but not for DST)
    // 'w': wall clock time (adjust for both time zone and DST offset)
    //
    // DST adjustment is done using the rule given as third argument
    var convertDateToUTC = function( date, type, rule ) {
      var offset = 0;

      if(type == 'u' || type == 'g' || type == 'z') { // UTC
          offset = 0;
      } else if(type == 's') { // Standard Time
          offset = basicOffset;
      } else if(type == 'w' || !type ) { // Wall Clock Time
          offset = getAdjustedOffset(basicOffset,rule);
      } else {
          throw("unknown type "+type);
      }
      offset *= 60*1000; // to millis

      return new Date( date.getTime() + offset );
    }

    // Step 1:  Find applicable rules for this year.
    // Step 2:  Sort the rules by effective date.
    // Step 3:  Check requested date to see if a rule has yet taken effect this year.  If not,
    // Step 4:  Get the rules for the previous year.  If there isn't an applicable rule for last year, then
    //      there probably is no current time offset since they seem to explicitly turn off the offset
    //      when someone stops observing DST.
    //      FIXME if this is not the case and we'll walk all the way back (ugh).
    // Step 5:  Sort the rules by effective date.
    // Step 6:  Apply the most recent rule before the current time.

    var convertRuleToExactDateAndTime = function( yearAndRule, prevRule )
    {
      var year = yearAndRule[0];
      var rule = yearAndRule[1];

      // Assume that the rule applies to the year of the given date.
      var months = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
      };

      var days = {
        "sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6
      }

      var hms = parseTimeString( rule[ 5 ] );
      var effectiveDate;

      if ( !isNaN( rule[ 4 ] ) ) // If we have a specific date, use that!
      {
        effectiveDate = new Date( Date.UTC( year, months[ rule[ 3 ] ], rule[ 4 ], hms[ 1 ], hms[ 2 ], hms[ 3 ], 0 ) );
      }
      else // Let's hunt for the date.
      {
        var targetDay,
          operator;

        if ( rule[ 4 ].substr( 0, 4 ) === "last" ) // Example: lastThu
        {
          // Start at the last day of the month and work backward.
          effectiveDate = new Date( Date.UTC( year, months[ rule[ 3 ] ] + 1, 1, hms[ 1 ] - 24, hms[ 2 ], hms[ 3 ], 0 ) );
          targetDay = days[ rule[ 4 ].substr( 4, 3 ).toLowerCase( ) ];
          operator = "<=";
        }
        else // Example: Sun>=15
        {
          // Start at the specified date.
          effectiveDate = new Date( Date.UTC( year, months[ rule[ 3 ] ], rule[ 4 ].substr( 5 ), hms[ 1 ], hms[ 2 ], hms[ 3 ], 0 ) );
          targetDay = days[ rule[ 4 ].substr( 0, 3 ).toLowerCase( ) ];
          operator = rule[ 4 ].substr( 3, 2 );
        }

        var ourDay = effectiveDate.getUTCDay( );

        if ( operator === ">=" ) // Go forwards.
        {
          effectiveDate.setUTCDate( effectiveDate.getUTCDate( ) + ( targetDay - ourDay + ( ( targetDay < ourDay ) ? 7 : 0 ) ) );
        }
        else // Go backwards.  Looking for the last of a certain day, or operator is "<=" (less likely).
        {
          effectiveDate.setUTCDate( effectiveDate.getUTCDate( ) + ( targetDay - ourDay - ( ( targetDay > ourDay ) ? 7 : 0 ) ) );
        }
      }

      // if previous rule is given, correct for the fact that the starting time of the current
      // rule may be specified in local time
      if(prevRule) {
        effectiveDate = convertDateToUTC(effectiveDate, hms[4], prevRule);
      }

      return effectiveDate;
    }

    var findApplicableRules = function( year, ruleset )
    {
      var applicableRules = [];

      for ( var i in ruleset )
      {
        if ( Number( ruleset[ i ][ 0 ] ) <= year ) // Exclude future rules.
        {
          if (
            Number( ruleset[ i ][ 1 ] ) >= year                                            // Date is in a set range.
            || ( Number( ruleset[ i ][ 0 ] ) === year && ruleset[ i ][ 1 ] === "only" )    // Date is in an "only" year.
            || ruleset[ i ][ 1 ] === "max"                                                 // We're in a range from the start year to infinity.
          )
          {
            // It's completely okay to have any number of matches here.
            // Normally we should only see two, but that doesn't preclude other numbers of matches.
            // These matches are applicable to this year.
            applicableRules.push( [year, ruleset[ i ]] );
          }
        }
      }

      return applicableRules;
    }

    var compareDates = function( a, b, prev )
    {
      if ( a.constructor !== Date ) {
        a = convertRuleToExactDateAndTime( a, prev );
      } else if(prev) {
        a = convertDateToUTC(a, isUTC?'u':'w', prev);
      }
      if ( b.constructor !== Date ) {
        b = convertRuleToExactDateAndTime( b, prev );
      } else if(prev) {
        b = convertDateToUTC(b, isUTC?'u':'w', prev);
      }

      a = Number( a );
      b = Number( b );

      return a - b;
    }

    var year = date.getUTCFullYear( );
    var applicableRules;

    applicableRules = findApplicableRules( year, _this.rules[ ruleset ] );
    applicableRules.push( date );
    // While sorting, the time zone in which the rule starting time is specified
    // is ignored. This is ok as long as the timespan between two DST changes is
    // larger than the DST offset, which is probably always true.
    // As the given date may indeed be close to a DST change, it may get sorted
    // to a wrong position (off by one), which is corrected below.
    applicableRules.sort( compareDates );

    if ( applicableRules.indexOf( date ) < 2 ) { // If there are not enough past DST rules...
      applicableRules = applicableRules.concat(findApplicableRules( year-1, _this.rules[ ruleset ] ));
      applicableRules.sort( compareDates );
    }

    var pinpoint = applicableRules.indexOf( date );
    if ( pinpoint > 1 && compareDates( date, applicableRules[pinpoint-1], applicableRules[pinpoint-2][1] ) < 0 ) {
      // the previous rule does not really apply, take the one before that
      return applicableRules[ pinpoint - 2 ][1];
    } else if ( pinpoint > 0 && pinpoint < applicableRules.length - 1 && compareDates( date, applicableRules[pinpoint+1], applicableRules[pinpoint-1][1] ) > 0) {
      // the next rule does already apply, take that one
      return applicableRules[ pinpoint + 1 ][1];
    } else if ( pinpoint === 0 ) {
      // no applicable rule found in this and in previous year
      return null;
    } else {
      return applicableRules[ pinpoint - 1 ][1];
    }
  }
  function getAdjustedOffset(off, rule) {
    var save = rule[6];
    var t = parseTimeString(save);
    var adj = save.indexOf('-') == 0 ? -1 : 1;
    var ret = (adj*(((t[1] *60 + t[2]) * 60 + t[3]) * 1000));
    ret = ret/60/1000;
    ret -= off
    ret = -Math.ceil(ret);
    return ret;
  }
  function getAbbreviation(zone, rule) {
    var res;
    var base = zone[2];
    if (base.indexOf('%s') > -1) {
      var repl;
      if (rule) {
        repl = rule[7]=='-'?'':rule[7];
      }
      // FIXME: Right now just falling back to Standard --
      // apparently ought to use the last valid rule,
      // although in practice that always ought to be Standard
      else {
        repl = 'S';
      }
      res = base.replace('%s', repl);
    }
    else if (base.indexOf('/') > -1) {
      // chose one of two alternative strings
      var t = parseTimeString(rule[6]);
      var isDst = (t[1])||(t[2])||(t[3]);
      res = base.split("/",2)[isDst?1:0];
    } else {
      res = base;
    }
    return res;
  }

  this.getTzInfo = function(dt, tz, isUTC) {
    var zone = getZone(dt, tz);
    var off = getBasicOffset(zone);
    // See if the offset needs adjustment
    var rule = getRule(dt, zone, isUTC);
    if (rule) {
      off = getAdjustedOffset(off, rule);
    }
    var abbr = getAbbreviation(zone, rule);
    return { tzOffset: off, tzAbbr: abbr };
  }
}

// Timezone data for: northamerica,europe
timezoneJS.timezone.zones = {"Europe/London":[["-0:01:15","-","LMT","1847","Dec","1","0:00s"],["0:00","GB-Eire","%s","1968","Oct","27"],["1:00","-","BST","1971","Oct","31","2:00u"],["0:00","GB-Eire","%s","1996"],["0:00","EU","GMT/BST"]],"Europe/Jersey":"Europe/London","Europe/Guernsey":"Europe/London","Europe/Isle_of_Man":"Europe/London","Europe/Dublin":[["-0:25:00","-","LMT","1880","Aug","2"],["-0:25:21","-","DMT","1916","May","21","2:00"],["-0:25:21","1:00","IST","1916","Oct","1","2:00s"],["0:00","GB-Eire","%s","1921","Dec","6",""],["0:00","GB-Eire","GMT/IST","1940","Feb","25","2:00"],["0:00","1:00","IST","1946","Oct","6","2:00"],["0:00","-","GMT","1947","Mar","16","2:00"],["0:00","1:00","IST","1947","Nov","2","2:00"],["0:00","-","GMT","1948","Apr","18","2:00"],["0:00","GB-Eire","GMT/IST","1968","Oct","27"],["1:00","-","IST","1971","Oct","31","2:00u"],["0:00","GB-Eire","GMT/IST","1996"],["0:00","EU","GMT/IST"]],"WET":[["0:00","EU","WE%sT"]],"CET":[["1:00","C-Eur","CE%sT"]],"MET":[["1:00","C-Eur","ME%sT"]],"EET":[["2:00","EU","EE%sT"]],"Europe/Tirane":[["1:19:20","-","LMT","1914"],["1:00","-","CET","1940","Jun","16"],["1:00","Albania","CE%sT","1984","Jul"],["1:00","EU","CE%sT"]],"Europe/Andorra":[["0:06:04","-","LMT","1901"],["0:00","-","WET","1946","Sep","30"],["1:00","-","CET","1985","Mar","31","2:00"],["1:00","EU","CE%sT"]],"Europe/Vienna":[["1:05:20","-","LMT","1893","Apr"],["1:00","C-Eur","CE%sT","1920"],["1:00","Austria","CE%sT","1940","Apr","1","2:00s"],["1:00","C-Eur","CE%sT","1945","Apr","2","2:00s"],["1:00","1:00","CEST","1945","Apr","12","2:00s"],["1:00","-","CET","1946"],["1:00","Austria","CE%sT","1981"],["1:00","EU","CE%sT"]],"Europe/Minsk":[["1:50:16","-","LMT","1880"],["1:50","-","MMT","1924","May","2",""],["2:00","-","EET","1930","Jun","21"],["3:00","-","MSK","1941","Jun","28"],["1:00","C-Eur","CE%sT","1944","Jul","3"],["3:00","Russia","MSK/MSD","1990"],["3:00","-","MSK","1991","Mar","31","2:00s"],["2:00","1:00","EEST","1991","Sep","29","2:00s"],["2:00","-","EET","1992","Mar","29","0:00s"],["2:00","1:00","EEST","1992","Sep","27","0:00s"],["2:00","Russia","EE%sT"]],"Europe/Brussels":[["0:17:30","-","LMT","1880"],["0:17:30","-","BMT","1892","May","1","12:00",""],["0:00","-","WET","1914","Nov","8"],["1:00","-","CET","1916","May","1","0:00"],["1:00","C-Eur","CE%sT","1918","Nov","11","11:00u"],["0:00","Belgium","WE%sT","1940","May","20","2:00s"],["1:00","C-Eur","CE%sT","1944","Sep","3"],["1:00","Belgium","CE%sT","1977"],["1:00","EU","CE%sT"]],"Europe/Sofia":[["1:33:16","-","LMT","1880"],["1:56:56","-","IMT","1894","Nov","30",""],["2:00","-","EET","1942","Nov","2","3:00"],["1:00","C-Eur","CE%sT","1945"],["1:00","-","CET","1945","Apr","2","3:00"],["2:00","-","EET","1979","Mar","31","23:00"],["2:00","Bulg","EE%sT","1982","Sep","26","2:00"],["2:00","C-Eur","EE%sT","1991"],["2:00","E-Eur","EE%sT","1997"],["2:00","EU","EE%sT"]],"Europe/Prague":[["0:57:44","-","LMT","1850"],["0:57:44","-","PMT","1891","Oct",""],["1:00","C-Eur","CE%sT","1944","Sep","17","2:00s"],["1:00","Czech","CE%sT","1979"],["1:00","EU","CE%sT"]],"Europe/Copenhagen":[["0:50:20","-","LMT","1890"],["0:50:20","-","CMT","1894","Jan","1",""],["1:00","Denmark","CE%sT","1942","Nov","2","2:00s"],["1:00","C-Eur","CE%sT","1945","Apr","2","2:00"],["1:00","Denmark","CE%sT","1980"],["1:00","EU","CE%sT"]],"Atlantic/Faroe":[["-0:27:04","-","LMT","1908","Jan","11",""],["0:00","-","WET","1981"],["0:00","EU","WE%sT"]],"America/Danmarkshavn":[["-1:14:40","-","LMT","1916","Jul","28"],["-3:00","-","WGT","1980","Apr","6","2:00"],["-3:00","EU","WG%sT","1996"],["0:00","-","GMT"]],"America/Scoresbysund":[["-1:27:52","-","LMT","1916","Jul","28",""],["-2:00","-","CGT","1980","Apr","6","2:00"],["-2:00","C-Eur","CG%sT","1981","Mar","29"],["-1:00","EU","EG%sT"]],"America/Godthab":[["-3:26:56","-","LMT","1916","Jul","28",""],["-3:00","-","WGT","1980","Apr","6","2:00"],["-3:00","EU","WG%sT"]],"America/Thule":[["-4:35:08","-","LMT","1916","Jul","28",""],["-4:00","Thule","A%sT"]],"Europe/Tallinn":[["1:39:00","-","LMT","1880"],["1:39:00","-","TMT","1918","Feb",""],["1:00","C-Eur","CE%sT","1919","Jul"],["1:39:00","-","TMT","1921","May"],["2:00","-","EET","1940","Aug","6"],["3:00","-","MSK","1941","Sep","15"],["1:00","C-Eur","CE%sT","1944","Sep","22"],["3:00","Russia","MSK/MSD","1989","Mar","26","2:00s"],["2:00","1:00","EEST","1989","Sep","24","2:00s"],["2:00","C-Eur","EE%sT","1998","Sep","22"],["2:00","EU","EE%sT","1999","Nov","1"],["2:00","-","EET","2002","Feb","21"],["2:00","EU","EE%sT"]],"Europe/Helsinki":[["1:39:52","-","LMT","1878","May","31"],["1:39:52","-","HMT","1921","May",""],["2:00","Finland","EE%sT","1983"],["2:00","EU","EE%sT"]],"Europe/Mariehamn":"Europe/Helsinki","Europe/Paris":[["0:09:21","-","LMT","1891","Mar","15","0:01"],["0:09:21","-","PMT","1911","Mar","11","0:01",""],["0:00","France","WE%sT","1940","Jun","14","23:00"],["1:00","C-Eur","CE%sT","1944","Aug","25"],["0:00","France","WE%sT","1945","Sep","16","3:00"],["1:00","France","CE%sT","1977"],["1:00","EU","CE%sT"]],"Europe/Berlin":[["0:53:28","-","LMT","1893","Apr"],["1:00","C-Eur","CE%sT","1945","May","24","2:00"],["1:00","SovietZone","CE%sT","1946"],["1:00","Germany","CE%sT","1980"],["1:00","EU","CE%sT"]],"Europe/Gibraltar":[["-0:21:24","-","LMT","1880","Aug","2","0:00s"],["0:00","GB-Eire","%s","1957","Apr","14","2:00"],["1:00","-","CET","1982"],["1:00","EU","CE%sT"]],"Europe/Athens":[["1:34:52","-","LMT","1895","Sep","14"],["1:34:52","-","AMT","1916","Jul","28","0:01",""],["2:00","Greece","EE%sT","1941","Apr","30"],["1:00","Greece","CE%sT","1944","Apr","4"],["2:00","Greece","EE%sT","1981"],[""],[""],["2:00","EU","EE%sT"]],"Europe/Budapest":[["1:16:20","-","LMT","1890","Oct"],["1:00","C-Eur","CE%sT","1918"],["1:00","Hungary","CE%sT","1941","Apr","6","2:00"],["1:00","C-Eur","CE%sT","1945"],["1:00","Hungary","CE%sT","1980","Sep","28","2:00s"],["1:00","EU","CE%sT"]],"Atlantic/Reykjavik":[["-1:27:24","-","LMT","1837"],["-1:27:48","-","RMT","1908",""],["-1:00","Iceland","IS%sT","1968","Apr","7","1:00s"],["0:00","-","GMT"]],"Europe/Rome":[["0:49:56","-","LMT","1866","Sep","22"],["0:49:56","-","RMT","1893","Nov","1","0:00s",""],["1:00","Italy","CE%sT","1942","Nov","2","2:00s"],["1:00","C-Eur","CE%sT","1944","Jul"],["1:00","Italy","CE%sT","1980"],["1:00","EU","CE%sT"]],"Europe/Vatican":"Europe/Rome","Europe/San_Marino":"Europe/Rome","Europe/Riga":[["1:36:24","-","LMT","1880"],["1:36:24","-","RMT","1918","Apr","15","2:00",""],["1:36:24","1:00","LST","1918","Sep","16","3:00",""],["1:36:24","-","RMT","1919","Apr","1","2:00"],["1:36:24","1:00","LST","1919","May","22","3:00"],["1:36:24","-","RMT","1926","May","11"],["2:00","-","EET","1940","Aug","5"],["3:00","-","MSK","1941","Jul"],["1:00","C-Eur","CE%sT","1944","Oct","13"],["3:00","Russia","MSK/MSD","1989","Mar","lastSun","2:00s"],["2:00","1:00","EEST","1989","Sep","lastSun","2:00s"],["2:00","Latvia","EE%sT","1997","Jan","21"],["2:00","EU","EE%sT","2000","Feb","29"],["2:00","-","EET","2001","Jan","2"],["2:00","EU","EE%sT"]],"Europe/Vaduz":[["0:38:04","-","LMT","1894","Jun"],["1:00","-","CET","1981"],["1:00","EU","CE%sT"]],"Europe/Vilnius":[["1:41:16","-","LMT","1880"],["1:24:00","-","WMT","1917",""],["1:35:36","-","KMT","1919","Oct","10",""],["1:00","-","CET","1920","Jul","12"],["2:00","-","EET","1920","Oct","9"],["1:00","-","CET","1940","Aug","3"],["3:00","-","MSK","1941","Jun","24"],["1:00","C-Eur","CE%sT","1944","Aug"],["3:00","Russia","MSK/MSD","1991","Mar","31","2:00s"],["2:00","1:00","EEST","1991","Sep","29","2:00s"],["2:00","C-Eur","EE%sT","1998"],["2:00","-","EET","1998","Mar","29","1:00u"],["1:00","EU","CE%sT","1999","Oct","31","1:00u"],["2:00","-","EET","2003","Jan","1"],["2:00","EU","EE%sT"]],"Europe/Luxembourg":[["0:24:36","-","LMT","1904","Jun"],["1:00","Lux","CE%sT","1918","Nov","25"],["0:00","Lux","WE%sT","1929","Oct","6","2:00s"],["0:00","Belgium","WE%sT","1940","May","14","3:00"],["1:00","C-Eur","WE%sT","1944","Sep","18","3:00"],["1:00","Belgium","CE%sT","1977"],["1:00","EU","CE%sT"]],"Europe/Malta":[["0:58:04","-","LMT","1893","Nov","2","0:00s",""],["1:00","Italy","CE%sT","1942","Nov","2","2:00s"],["1:00","C-Eur","CE%sT","1945","Apr","2","2:00s"],["1:00","Italy","CE%sT","1973","Mar","31"],["1:00","Malta","CE%sT","1981"],["1:00","EU","CE%sT"]],"Europe/Chisinau":[["1:55:20","-","LMT","1880"],["1:55","-","CMT","1918","Feb","15",""],["1:44:24","-","BMT","1931","Jul","24",""],["2:00","Romania","EE%sT","1940","Aug","15"],["2:00","1:00","EEST","1941","Jul","17"],["1:00","C-Eur","CE%sT","1944","Aug","24"],["3:00","Russia","MSK/MSD","1990"],["3:00","-","MSK","1990","May","6"],["2:00","-","EET","1991"],["2:00","Russia","EE%sT","1992"],["2:00","E-Eur","EE%sT","1997"],["2:00","EU","EE%sT"]],"Europe/Monaco":[["0:29:32","-","LMT","1891","Mar","15"],["0:09:21","-","PMT","1911","Mar","11",""],["0:00","France","WE%sT","1945","Sep","16","3:00"],["1:00","France","CE%sT","1977"],["1:00","EU","CE%sT"]],"Europe/Amsterdam":[["0:19:32","-","LMT","1835"],["0:19:32","Neth","%s","1937","Jul","1"],["0:20","Neth","NE%sT","1940","May","16","0:00",""],["1:00","C-Eur","CE%sT","1945","Apr","2","2:00"],["1:00","Neth","CE%sT","1977"],["1:00","EU","CE%sT"]],"Europe/Oslo":[["0:43:00","-","LMT","1895","Jan","1"],["1:00","Norway","CE%sT","1940","Aug","10","23:00"],["1:00","C-Eur","CE%sT","1945","Apr","2","2:00"],["1:00","Norway","CE%sT","1980"],["1:00","EU","CE%sT"]],"Arctic/Longyearbyen":"Europe/Oslo","Europe/Warsaw":[["1:24:00","-","LMT","1880"],["1:24:00","-","WMT","1915","Aug","5",""],["1:00","C-Eur","CE%sT","1918","Sep","16","3:00"],["2:00","Poland","EE%sT","1922","Jun"],["1:00","Poland","CE%sT","1940","Jun","23","2:00"],["1:00","C-Eur","CE%sT","1944","Oct"],["1:00","Poland","CE%sT","1977"],["1:00","W-Eur","CE%sT","1988"],["1:00","EU","CE%sT"]],"Europe/Lisbon":[["-0:36:32","-","LMT","1884"],["-0:36:32","-","LMT","1912","Jan","1",""],["0:00","Port","WE%sT","1966","Apr","3","2:00"],["1:00","-","CET","1976","Sep","26","1:00"],["0:00","Port","WE%sT","1983","Sep","25","1:00s"],["0:00","W-Eur","WE%sT","1992","Sep","27","1:00s"],["1:00","EU","CE%sT","1996","Mar","31","1:00u"],["0:00","EU","WE%sT"]],"Atlantic/Azores":[["-1:42:40","-","LMT","1884",""],["-1:54:32","-","HMT","1911","May","24",""],["-2:00","Port","AZO%sT","1966","Apr","3","2:00",""],["-1:00","Port","AZO%sT","1983","Sep","25","1:00s"],["-1:00","W-Eur","AZO%sT","1992","Sep","27","1:00s"],["0:00","EU","WE%sT","1993","Mar","28","1:00u"],["-1:00","EU","AZO%sT"]],"Atlantic/Madeira":[["-1:07:36","-","LMT","1884",""],["-1:07:36","-","FMT","1911","May","24",""],["-1:00","Port","MAD%sT","1966","Apr","3","2:00",""],["0:00","Port","WE%sT","1983","Sep","25","1:00s"],["0:00","EU","WE%sT"]],"Europe/Bucharest":[["1:44:24","-","LMT","1891","Oct"],["1:44:24","-","BMT","1931","Jul","24",""],["2:00","Romania","EE%sT","1981","Mar","29","2:00s"],["2:00","C-Eur","EE%sT","1991"],["2:00","Romania","EE%sT","1994"],["2:00","E-Eur","EE%sT","1997"],["2:00","EU","EE%sT"]],"Europe/Kaliningrad":[["1:22:00","-","LMT","1893","Apr"],["1:00","C-Eur","CE%sT","1945"],["2:00","Poland","CE%sT","1946"],["3:00","Russia","MSK/MSD","1991","Mar","31","2:00s"],["2:00","Russia","EE%sT","2011","Mar","27","2:00s"],["3:00","-","EET"]],"Europe/Moscow":[["2:30:20","-","LMT","1880"],["2:30","-","MMT","1916","Jul","3",""],["2:30:48","Russia","%s","1919","Jul","1","2:00"],["3:00","Russia","MSK/MSD","1922","Oct"],["2:00","-","EET","1930","Jun","21"],["3:00","Russia","MSK/MSD","1991","Mar","31","2:00s"],["2:00","Russia","EE%sT","1992","Jan","19","2:00s"],["3:00","Russia","MSK/MSD","2011","Mar","27","2:00s"],["4:00","-","MSK"]],"Europe/Volgograd":[["2:57:40","-","LMT","1920","Jan","3"],["3:00","-","TSAT","1925","Apr","6",""],["3:00","-","STAT","1930","Jun","21",""],["4:00","-","STAT","1961","Nov","11"],["4:00","Russia","VOL%sT","1989","Mar","26","2:00s",""],["3:00","Russia","VOL%sT","1991","Mar","31","2:00s"],["4:00","-","VOLT","1992","Mar","29","2:00s"],["3:00","Russia","VOL%sT","2011","Mar","27","2:00s"],["4:00","-","VOLT"]],"Europe/Samara":[["3:20:36","-","LMT","1919","Jul","1","2:00"],["3:00","-","SAMT","1930","Jun","21"],["4:00","-","SAMT","1935","Jan","27"],["4:00","Russia","KUY%sT","1989","Mar","26","2:00s",""],["3:00","Russia","KUY%sT","1991","Mar","31","2:00s"],["2:00","Russia","KUY%sT","1991","Sep","29","2:00s"],["3:00","-","KUYT","1991","Oct","20","3:00"],["4:00","Russia","SAM%sT","2010","Mar","28","2:00s",""],["3:00","Russia","SAM%sT","2011","Mar","27","2:00s"],["4:00","-","SAMT"]],"Asia/Yekaterinburg":[["4:02:24","-","LMT","1919","Jul","15","4:00"],["4:00","-","SVET","1930","Jun","21",""],["5:00","Russia","SVE%sT","1991","Mar","31","2:00s"],["4:00","Russia","SVE%sT","1992","Jan","19","2:00s"],["5:00","Russia","YEK%sT","2011","Mar","27","2:00s"],["6:00","-","YEKT",""]],"Asia/Omsk":[["4:53:36","-","LMT","1919","Nov","14"],["5:00","-","OMST","1930","Jun","21",""],["6:00","Russia","OMS%sT","1991","Mar","31","2:00s"],["5:00","Russia","OMS%sT","1992","Jan","19","2:00s"],["6:00","Russia","OMS%sT","2011","Mar","27","2:00s"],["7:00","-","OMST"]],"Asia/Novosibirsk":[["5:31:40","-","LMT","1919","Dec","14","6:00"],["6:00","-","NOVT","1930","Jun","21",""],["7:00","Russia","NOV%sT","1991","Mar","31","2:00s"],["6:00","Russia","NOV%sT","1992","Jan","19","2:00s"],["7:00","Russia","NOV%sT","1993","May","23",""],["6:00","Russia","NOV%sT","2011","Mar","27","2:00s"],["7:00","-","NOVT"]],"Asia/Novokuznetsk":[["5:48:48","-","NMT","1920","Jan","6"],["6:00","-","KRAT","1930","Jun","21",""],["7:00","Russia","KRA%sT","1991","Mar","31","2:00s"],["6:00","Russia","KRA%sT","1992","Jan","19","2:00s"],["7:00","Russia","KRA%sT","2010","Mar","28","2:00s"],["6:00","Russia","NOV%sT","2011","Mar","27","2:00s"],["7:00","-","NOVT",""]],"Asia/Krasnoyarsk":[["6:11:20","-","LMT","1920","Jan","6"],["6:00","-","KRAT","1930","Jun","21",""],["7:00","Russia","KRA%sT","1991","Mar","31","2:00s"],["6:00","Russia","KRA%sT","1992","Jan","19","2:00s"],["7:00","Russia","KRA%sT","2011","Mar","27","2:00s"],["8:00","-","KRAT"]],"Asia/Irkutsk":[["6:57:20","-","LMT","1880"],["6:57:20","-","IMT","1920","Jan","25",""],["7:00","-","IRKT","1930","Jun","21",""],["8:00","Russia","IRK%sT","1991","Mar","31","2:00s"],["7:00","Russia","IRK%sT","1992","Jan","19","2:00s"],["8:00","Russia","IRK%sT","2011","Mar","27","2:00s"],["9:00","-","IRKT"]],"Asia/Yakutsk":[["8:38:40","-","LMT","1919","Dec","15"],["8:00","-","YAKT","1930","Jun","21",""],["9:00","Russia","YAK%sT","1991","Mar","31","2:00s"],["8:00","Russia","YAK%sT","1992","Jan","19","2:00s"],["9:00","Russia","YAK%sT","2011","Mar","27","2:00s"],["10:00","-","YAKT"]],"Asia/Vladivostok":[["8:47:44","-","LMT","1922","Nov","15"],["9:00","-","VLAT","1930","Jun","21",""],["10:00","Russia","VLA%sT","1991","Mar","31","2:00s"],["9:00","Russia","VLA%sST","1992","Jan","19","2:00s"],["10:00","Russia","VLA%sT","2011","Mar","27","2:00s"],["11:00","-","VLAT"]],"Asia/Sakhalin":[["9:30:48","-","LMT","1905","Aug","23"],["9:00","-","CJT","1938"],["9:00","-","JST","1945","Aug","25"],["11:00","Russia","SAK%sT","1991","Mar","31","2:00s",""],["10:00","Russia","SAK%sT","1992","Jan","19","2:00s"],["11:00","Russia","SAK%sT","1997","Mar","lastSun","2:00s"],["10:00","Russia","SAK%sT","2011","Mar","27","2:00s"],["11:00","-","SAKT"]],"Asia/Magadan":[["10:03:12","-","LMT","1924","May","2"],["10:00","-","MAGT","1930","Jun","21",""],["11:00","Russia","MAG%sT","1991","Mar","31","2:00s"],["10:00","Russia","MAG%sT","1992","Jan","19","2:00s"],["11:00","Russia","MAG%sT","2011","Mar","27","2:00s"],["12:00","-","MAGT"]],"Asia/Kamchatka":[["10:34:36","-","LMT","1922","Nov","10"],["11:00","-","PETT","1930","Jun","21",""],["12:00","Russia","PET%sT","1991","Mar","31","2:00s"],["11:00","Russia","PET%sT","1992","Jan","19","2:00s"],["12:00","Russia","PET%sT","2010","Mar","28","2:00s"],["11:00","Russia","PET%sT","2011","Mar","27","2:00s"],["12:00","-","PETT"]],"Asia/Anadyr":[["11:49:56","-","LMT","1924","May","2"],["12:00","-","ANAT","1930","Jun","21",""],["13:00","Russia","ANA%sT","1982","Apr","1","0:00s"],["12:00","Russia","ANA%sT","1991","Mar","31","2:00s"],["11:00","Russia","ANA%sT","1992","Jan","19","2:00s"],["12:00","Russia","ANA%sT","2010","Mar","28","2:00s"],["11:00","Russia","ANA%sT","2011","Mar","27","2:00s"],["12:00","-","ANAT"]],"Europe/Belgrade":[["1:22:00","-","LMT","1884"],["1:00","-","CET","1941","Apr","18","23:00"],["1:00","C-Eur","CE%sT","1945"],["1:00","-","CET","1945","May","8","2:00s"],["1:00","1:00","CEST","1945","Sep","16","2:00s"],["1:00","-","CET","1982","Nov","27"],["1:00","EU","CE%sT"]],"Europe/Ljubljana":"Europe/Belgrade","Europe/Podgorica":"Europe/Belgrade","Europe/Sarajevo":"Europe/Belgrade","Europe/Skopje":"Europe/Belgrade","Europe/Zagreb":"Europe/Belgrade","Europe/Bratislava":"Europe/Prague","Europe/Madrid":[["-0:14:44","-","LMT","1901","Jan","1","0:00s"],["0:00","Spain","WE%sT","1946","Sep","30"],["1:00","Spain","CE%sT","1979"],["1:00","EU","CE%sT"]],"Africa/Ceuta":[["-0:21:16","-","LMT","1901"],["0:00","-","WET","1918","May","6","23:00"],["0:00","1:00","WEST","1918","Oct","7","23:00"],["0:00","-","WET","1924"],["0:00","Spain","WE%sT","1929"],["0:00","SpainAfrica","WE%sT","1984","Mar","16"],["1:00","-","CET","1986"],["1:00","EU","CE%sT"]],"Atlantic/Canary":[["-1:01:36","-","LMT","1922","Mar",""],["-1:00","-","CANT","1946","Sep","30","1:00",""],["0:00","-","WET","1980","Apr","6","0:00s"],["0:00","1:00","WEST","1980","Sep","28","0:00s"],["0:00","EU","WE%sT"]],"Europe/Stockholm":[["1:12:12","-","LMT","1879","Jan","1"],["1:00:14","-","SET","1900","Jan","1",""],["1:00","-","CET","1916","May","14","23:00"],["1:00","1:00","CEST","1916","Oct","1","01:00"],["1:00","-","CET","1980"],["1:00","EU","CE%sT"]],"Europe/Zurich":[["0:34:08","-","LMT","1848","Sep","12"],["0:29:44","-","BMT","1894","Jun",""],["1:00","Swiss","CE%sT","1981"],["1:00","EU","CE%sT"]],"Europe/Istanbul":[["1:55:52","-","LMT","1880"],["1:56:56","-","IMT","1910","Oct",""],["2:00","Turkey","EE%sT","1978","Oct","15"],["3:00","Turkey","TR%sT","1985","Apr","20",""],["2:00","Turkey","EE%sT","2007"],["2:00","EU","EE%sT","2011","Mar","27","1:00u"],["2:00","-","EET","2011","Mar","28","1:00u"],["2:00","EU","EE%sT"]],"Asia/Istanbul":"Europe/Istanbul","Europe/Kiev":[["2:02:04","-","LMT","1880"],["2:02:04","-","KMT","1924","May","2",""],["2:00","-","EET","1930","Jun","21"],["3:00","-","MSK","1941","Sep","20"],["1:00","C-Eur","CE%sT","1943","Nov","6"],["3:00","Russia","MSK/MSD","1990"],["3:00","-","MSK","1990","Jul","1","2:00"],["2:00","-","EET","1992"],["2:00","E-Eur","EE%sT","1995"],["2:00","EU","EE%sT"]],"Europe/Uzhgorod":[["1:29:12","-","LMT","1890","Oct"],["1:00","-","CET","1940"],["1:00","C-Eur","CE%sT","1944","Oct"],["1:00","1:00","CEST","1944","Oct","26"],["1:00","-","CET","1945","Jun","29"],["3:00","Russia","MSK/MSD","1990"],["3:00","-","MSK","1990","Jul","1","2:00"],["1:00","-","CET","1991","Mar","31","3:00"],["2:00","-","EET","1992"],["2:00","E-Eur","EE%sT","1995"],["2:00","EU","EE%sT"]],"Europe/Zaporozhye":[["2:20:40","-","LMT","1880"],["2:20","-","CUT","1924","May","2",""],["2:00","-","EET","1930","Jun","21"],["3:00","-","MSK","1941","Aug","25"],["1:00","C-Eur","CE%sT","1943","Oct","25"],["3:00","Russia","MSK/MSD","1991","Mar","31","2:00"],["2:00","E-Eur","EE%sT","1995"],["2:00","EU","EE%sT"]],"Europe/Simferopol":[["2:16:24","-","LMT","1880"],["2:16","-","SMT","1924","May","2",""],["2:00","-","EET","1930","Jun","21"],["3:00","-","MSK","1941","Nov"],["1:00","C-Eur","CE%sT","1944","Apr","13"],["3:00","Russia","MSK/MSD","1990"],["3:00","-","MSK","1990","Jul","1","2:00"],["2:00","-","EET","1992"],["2:00","E-Eur","EE%sT","1994","May"],["3:00","E-Eur","MSK/MSD","1996","Mar","31","3:00s"],["3:00","1:00","MSD","1996","Oct","27","3:00s"],["3:00","Russia","MSK/MSD","1997"],["3:00","-","MSK","1997","Mar","lastSun","1:00u"],["2:00","EU","EE%sT"]],"EST":[["-5:00","-","EST"]],"MST":[["-7:00","-","MST"]],"HST":[["-10:00","-","HST"]],"EST5EDT":[["-5:00","US","E%sT"]],"CST6CDT":[["-6:00","US","C%sT"]],"MST7MDT":[["-7:00","US","M%sT"]],"PST8PDT":[["-8:00","US","P%sT"]],"America/New_York":[["-4:56:02","-","LMT","1883","Nov","18","12:03:58"],["-5:00","US","E%sT","1920"],["-5:00","NYC","E%sT","1942"],["-5:00","US","E%sT","1946"],["-5:00","NYC","E%sT","1967"],["-5:00","US","E%sT"]],"America/Chicago":[["-5:50:36","-","LMT","1883","Nov","18","12:09:24"],["-6:00","US","C%sT","1920"],["-6:00","Chicago","C%sT","1936","Mar","1","2:00"],["-5:00","-","EST","1936","Nov","15","2:00"],["-6:00","Chicago","C%sT","1942"],["-6:00","US","C%sT","1946"],["-6:00","Chicago","C%sT","1967"],["-6:00","US","C%sT"]],"America/North_Dakota/Center":[["-6:45:12","-","LMT","1883","Nov","18","12:14:48"],["-7:00","US","M%sT","1992","Oct","25","02:00"],["-6:00","US","C%sT"]],"America/North_Dakota/New_Salem":[["-6:45:39","-","LMT","1883","Nov","18","12:14:21"],["-7:00","US","M%sT","2003","Oct","26","02:00"],["-6:00","US","C%sT"]],"America/North_Dakota/Beulah":[["-6:47:07","-","LMT","1883","Nov","18","12:12:53"],["-7:00","US","M%sT","2010","Nov","7","2:00"],["-6:00","US","C%sT"]],"America/Denver":[["-6:59:56","-","LMT","1883","Nov","18","12:00:04"],["-7:00","US","M%sT","1920"],["-7:00","Denver","M%sT","1942"],["-7:00","US","M%sT","1946"],["-7:00","Denver","M%sT","1967"],["-7:00","US","M%sT"]],"America/Los_Angeles":[["-7:52:58","-","LMT","1883","Nov","18","12:07:02"],["-8:00","US","P%sT","1946"],["-8:00","CA","P%sT","1967"],["-8:00","US","P%sT"]],"America/Juneau":[["15:02:19","-","LMT","1867","Oct","18"],["-8:57:41","-","LMT","1900","Aug","20","12:00"],["-8:00","-","PST","1942"],["-8:00","US","P%sT","1946"],["-8:00","-","PST","1969"],["-8:00","US","P%sT","1980","Apr","27","2:00"],["-9:00","US","Y%sT","1980","Oct","26","2:00",""],["-8:00","US","P%sT","1983","Oct","30","2:00"],["-9:00","US","Y%sT","1983","Nov","30"],["-9:00","US","AK%sT"]],"America/Sitka":[["-14:58:47","-","LMT","1867","Oct","18"],["-9:01:13","-","LMT","1900","Aug","20","12:00"],["-8:00","-","PST","1942"],["-8:00","US","P%sT","1946"],["-8:00","-","PST","1969"],["-8:00","US","P%sT","1983","Oct","30","2:00"],["-9:00","US","Y%sT","1983","Nov","30"],["-9:00","US","AK%sT"]],"America/Metlakatla":[["15:13:42","-","LMT","1867","Oct","18"],["-8:46:18","-","LMT","1900","Aug","20","12:00"],["-8:00","-","PST","1942"],["-8:00","US","P%sT","1946"],["-8:00","-","PST","1969"],["-8:00","US","P%sT","1983","Oct","30","2:00"],["-8:00","US","MeST"]],"America/Yakutat":[["14:41:05","-","LMT","1867","Oct","18"],["-9:18:55","-","LMT","1900","Aug","20","12:00"],["-9:00","-","YST","1942"],["-9:00","US","Y%sT","1946"],["-9:00","-","YST","1969"],["-9:00","US","Y%sT","1983","Nov","30"],["-9:00","US","AK%sT"]],"America/Anchorage":[["14:00:24","-","LMT","1867","Oct","18"],["-9:59:36","-","LMT","1900","Aug","20","12:00"],["-10:00","-","CAT","1942"],["-10:00","US","CAT/CAWT","1945","Aug","14","23:00u"],["-10:00","US","CAT/CAPT","1946",""],["-10:00","-","CAT","1967","Apr"],["-10:00","-","AHST","1969"],["-10:00","US","AH%sT","1983","Oct","30","2:00"],["-9:00","US","Y%sT","1983","Nov","30"],["-9:00","US","AK%sT"]],"America/Nome":[["12:58:21","-","LMT","1867","Oct","18"],["-11:01:38","-","LMT","1900","Aug","20","12:00"],["-11:00","-","NST","1942"],["-11:00","US","N%sT","1946"],["-11:00","-","NST","1967","Apr"],["-11:00","-","BST","1969"],["-11:00","US","B%sT","1983","Oct","30","2:00"],["-9:00","US","Y%sT","1983","Nov","30"],["-9:00","US","AK%sT"]],"America/Adak":[["12:13:21","-","LMT","1867","Oct","18"],["-11:46:38","-","LMT","1900","Aug","20","12:00"],["-11:00","-","NST","1942"],["-11:00","US","N%sT","1946"],["-11:00","-","NST","1967","Apr"],["-11:00","-","BST","1969"],["-11:00","US","B%sT","1983","Oct","30","2:00"],["-10:00","US","AH%sT","1983","Nov","30"],["-10:00","US","HA%sT"]],"Pacific/Honolulu":[["-10:31:26","-","LMT","1896","Jan","13","12:00",""],["-10:30","-","HST","1933","Apr","30","2:00",""],["-10:30","1:00","HDT","1933","May","21","12:00",""],["-10:30","-","HST","1942","Feb","09","2:00",""],["-10:30","1:00","HDT","1945","Sep","30","2:00",""],["-10:30","US","H%sT","1947","Jun","8","2:00",""],["-10:00","-","HST"]],"America/Phoenix":[["-7:28:18","-","LMT","1883","Nov","18","11:31:42"],["-7:00","US","M%sT","1944","Jan","1","00:01"],["-7:00","-","MST","1944","Apr","1","00:01"],["-7:00","US","M%sT","1944","Oct","1","00:01"],["-7:00","-","MST","1967"],["-7:00","US","M%sT","1968","Mar","21"],["-7:00","-","MST"]],"America/Shiprock":"America/Denver","America/Boise":[["-7:44:49","-","LMT","1883","Nov","18","12:15:11"],["-8:00","US","P%sT","1923","May","13","2:00"],["-7:00","US","M%sT","1974"],["-7:00","-","MST","1974","Feb","3","2:00"],["-7:00","US","M%sT"]],"America/Indiana/Indianapolis":[["-5:44:38","-","LMT","1883","Nov","18","12:15:22"],["-6:00","US","C%sT","1920"],["-6:00","Indianapolis","C%sT","1942"],["-6:00","US","C%sT","1946"],["-6:00","Indianapolis","C%sT","1955","Apr","24","2:00"],["-5:00","-","EST","1957","Sep","29","2:00"],["-6:00","-","CST","1958","Apr","27","2:00"],["-5:00","-","EST","1969"],["-5:00","US","E%sT","1971"],["-5:00","-","EST","2006"],["-5:00","US","E%sT"]],"America/Indiana/Marengo":[["-5:45:23","-","LMT","1883","Nov","18","12:14:37"],["-6:00","US","C%sT","1951"],["-6:00","Marengo","C%sT","1961","Apr","30","2:00"],["-5:00","-","EST","1969"],["-5:00","US","E%sT","1974","Jan","6","2:00"],["-6:00","1:00","CDT","1974","Oct","27","2:00"],["-5:00","US","E%sT","1976"],["-5:00","-","EST","2006"],["-5:00","US","E%sT"]],"America/Indiana/Vincennes":[["-5:50:07","-","LMT","1883","Nov","18","12:09:53"],["-6:00","US","C%sT","1946"],["-6:00","Vincennes","C%sT","1964","Apr","26","2:00"],["-5:00","-","EST","1969"],["-5:00","US","E%sT","1971"],["-5:00","-","EST","2006","Apr","2","2:00"],["-6:00","US","C%sT","2007","Nov","4","2:00"],["-5:00","US","E%sT"]],"America/Indiana/Tell_City":[["-5:47:03","-","LMT","1883","Nov","18","12:12:57"],["-6:00","US","C%sT","1946"],["-6:00","Perry","C%sT","1964","Apr","26","2:00"],["-5:00","-","EST","1969"],["-5:00","US","E%sT","1971"],["-5:00","-","EST","2006","Apr","2","2:00"],["-6:00","US","C%sT"]],"America/Indiana/Petersburg":[["-5:49:07","-","LMT","1883","Nov","18","12:10:53"],["-6:00","US","C%sT","1955"],["-6:00","Pike","C%sT","1965","Apr","25","2:00"],["-5:00","-","EST","1966","Oct","30","2:00"],["-6:00","US","C%sT","1977","Oct","30","2:00"],["-5:00","-","EST","2006","Apr","2","2:00"],["-6:00","US","C%sT","2007","Nov","4","2:00"],["-5:00","US","E%sT"]],"America/Indiana/Knox":[["-5:46:30","-","LMT","1883","Nov","18","12:13:30"],["-6:00","US","C%sT","1947"],["-6:00","Starke","C%sT","1962","Apr","29","2:00"],["-5:00","-","EST","1963","Oct","27","2:00"],["-6:00","US","C%sT","1991","Oct","27","2:00"],["-5:00","-","EST","2006","Apr","2","2:00"],["-6:00","US","C%sT"]],"America/Indiana/Winamac":[["-5:46:25","-","LMT","1883","Nov","18","12:13:35"],["-6:00","US","C%sT","1946"],["-6:00","Pulaski","C%sT","1961","Apr","30","2:00"],["-5:00","-","EST","1969"],["-5:00","US","E%sT","1971"],["-5:00","-","EST","2006","Apr","2","2:00"],["-6:00","US","C%sT","2007","Mar","11","2:00"],["-5:00","US","E%sT"]],"America/Indiana/Vevay":[["-5:40:16","-","LMT","1883","Nov","18","12:19:44"],["-6:00","US","C%sT","1954","Apr","25","2:00"],["-5:00","-","EST","1969"],["-5:00","US","E%sT","1973"],["-5:00","-","EST","2006"],["-5:00","US","E%sT"]],"America/Kentucky/Louisville":[["-5:43:02","-","LMT","1883","Nov","18","12:16:58"],["-6:00","US","C%sT","1921"],["-6:00","Louisville","C%sT","1942"],["-6:00","US","C%sT","1946"],["-6:00","Louisville","C%sT","1961","Jul","23","2:00"],["-5:00","-","EST","1968"],["-5:00","US","E%sT","1974","Jan","6","2:00"],["-6:00","1:00","CDT","1974","Oct","27","2:00"],["-5:00","US","E%sT"]],"America/Kentucky/Monticello":[["-5:39:24","-","LMT","1883","Nov","18","12:20:36"],["-6:00","US","C%sT","1946"],["-6:00","-","CST","1968"],["-6:00","US","C%sT","2000","Oct","29","2:00"],["-5:00","US","E%sT"]],"America/Detroit":[["-5:32:11","-","LMT","1905"],["-6:00","-","CST","1915","May","15","2:00"],["-5:00","-","EST","1942"],["-5:00","US","E%sT","1946"],["-5:00","Detroit","E%sT","1973"],["-5:00","US","E%sT","1975"],["-5:00","-","EST","1975","Apr","27","2:00"],["-5:00","US","E%sT"]],"America/Menominee":[["-5:50:27","-","LMT","1885","Sep","18","12:00"],["-6:00","US","C%sT","1946"],["-6:00","Menominee","C%sT","1969","Apr","27","2:00"],["-5:00","-","EST","1973","Apr","29","2:00"],["-6:00","US","C%sT"]],"America/St_Johns":[["-3:30:52","-","LMT","1884"],["-3:30:52","StJohns","N%sT","1918"],["-3:30:52","Canada","N%sT","1919"],["-3:30:52","StJohns","N%sT","1935","Mar","30"],["-3:30","StJohns","N%sT","1942","May","11"],["-3:30","Canada","N%sT","1946"],["-3:30","StJohns","N%sT"]],"America/Goose_Bay":[["-4:01:40","-","LMT","1884",""],["-3:30:52","-","NST","1918"],["-3:30:52","Canada","N%sT","1919"],["-3:30:52","-","NST","1935","Mar","30"],["-3:30","-","NST","1936"],["-3:30","StJohns","N%sT","1942","May","11"],["-3:30","Canada","N%sT","1946"],["-3:30","StJohns","N%sT","1966","Mar","15","2:00"],["-4:00","StJohns","A%sT"]],"America/Halifax":[["-4:14:24","-","LMT","1902","Jun","15"],["-4:00","Halifax","A%sT","1918"],["-4:00","Canada","A%sT","1919"],["-4:00","Halifax","A%sT","1942","Feb","9","2:00s"],["-4:00","Canada","A%sT","1946"],["-4:00","Halifax","A%sT","1974"],["-4:00","Canada","A%sT"]],"America/Glace_Bay":[["-3:59:48","-","LMT","1902","Jun","15"],["-4:00","Canada","A%sT","1953"],["-4:00","Halifax","A%sT","1954"],["-4:00","-","AST","1972"],["-4:00","Halifax","A%sT","1974"],["-4:00","Canada","A%sT"]],"America/Moncton":[["-4:19:08","-","LMT","1883","Dec","9"],["-5:00","-","EST","1902","Jun","15"],["-4:00","Canada","A%sT","1933"],["-4:00","Moncton","A%sT","1942"],["-4:00","Canada","A%sT","1946"],["-4:00","Moncton","A%sT","1973"],["-4:00","Canada","A%sT","1993"],["-4:00","Moncton","A%sT","2007"],["-4:00","Canada","A%sT"]],"America/Blanc-Sablon":[["-3:48:28","-","LMT","1884"],["-4:00","Canada","A%sT","1970"],["-4:00","-","AST"]],"America/Montreal":[["-4:54:16","-","LMT","1884"],["-5:00","Mont","E%sT","1918"],["-5:00","Canada","E%sT","1919"],["-5:00","Mont","E%sT","1942","Feb","9","2:00s"],["-5:00","Canada","E%sT","1946"],["-5:00","Mont","E%sT","1974"],["-5:00","Canada","E%sT"]],"America/Toronto":[["-5:17:32","-","LMT","1895"],["-5:00","Canada","E%sT","1919"],["-5:00","Toronto","E%sT","1942","Feb","9","2:00s"],["-5:00","Canada","E%sT","1946"],["-5:00","Toronto","E%sT","1974"],["-5:00","Canada","E%sT"]],"America/Thunder_Bay":[["-5:57:00","-","LMT","1895"],["-6:00","-","CST","1910"],["-5:00","-","EST","1942"],["-5:00","Canada","E%sT","1970"],["-5:00","Mont","E%sT","1973"],["-5:00","-","EST","1974"],["-5:00","Canada","E%sT"]],"America/Nipigon":[["-5:53:04","-","LMT","1895"],["-5:00","Canada","E%sT","1940","Sep","29"],["-5:00","1:00","EDT","1942","Feb","9","2:00s"],["-5:00","Canada","E%sT"]],"America/Rainy_River":[["-6:18:16","-","LMT","1895"],["-6:00","Canada","C%sT","1940","Sep","29"],["-6:00","1:00","CDT","1942","Feb","9","2:00s"],["-6:00","Canada","C%sT"]],"America/Atikokan":[["-6:06:28","-","LMT","1895"],["-6:00","Canada","C%sT","1940","Sep","29"],["-6:00","1:00","CDT","1942","Feb","9","2:00s"],["-6:00","Canada","C%sT","1945","Sep","30","2:00"],["-5:00","-","EST"]],"America/Winnipeg":[["-6:28:36","-","LMT","1887","Jul","16"],["-6:00","Winn","C%sT","2006"],["-6:00","Canada","C%sT"]],"America/Regina":[["-6:58:36","-","LMT","1905","Sep"],["-7:00","Regina","M%sT","1960","Apr","lastSun","2:00"],["-6:00","-","CST"]],"America/Swift_Current":[["-7:11:20","-","LMT","1905","Sep"],["-7:00","Canada","M%sT","1946","Apr","lastSun","2:00"],["-7:00","Regina","M%sT","1950"],["-7:00","Swift","M%sT","1972","Apr","lastSun","2:00"],["-6:00","-","CST"]],"America/Edmonton":[["-7:33:52","-","LMT","1906","Sep"],["-7:00","Edm","M%sT","1987"],["-7:00","Canada","M%sT"]],"America/Vancouver":[["-8:12:28","-","LMT","1884"],["-8:00","Vanc","P%sT","1987"],["-8:00","Canada","P%sT"]],"America/Dawson_Creek":[["-8:00:56","-","LMT","1884"],["-8:00","Canada","P%sT","1947"],["-8:00","Vanc","P%sT","1972","Aug","30","2:00"],["-7:00","-","MST"]],"America/Pangnirtung":[["0","-","zzz","1921",""],["-4:00","NT_YK","A%sT","1995","Apr","Sun>=1","2:00"],["-5:00","Canada","E%sT","1999","Oct","31","2:00"],["-6:00","Canada","C%sT","2000","Oct","29","2:00"],["-5:00","Canada","E%sT"]],"America/Iqaluit":[["0","-","zzz","1942","Aug",""],["-5:00","NT_YK","E%sT","1999","Oct","31","2:00"],["-6:00","Canada","C%sT","2000","Oct","29","2:00"],["-5:00","Canada","E%sT"]],"America/Resolute":[["0","-","zzz","1947","Aug","31",""],["-6:00","NT_YK","C%sT","2000","Oct","29","2:00"],["-5:00","-","EST","2001","Apr","1","3:00"],["-6:00","Canada","C%sT","2006","Oct","29","2:00"],["-5:00","Resolute","%sT"]],"America/Rankin_Inlet":[["0","-","zzz","1957",""],["-6:00","NT_YK","C%sT","2000","Oct","29","2:00"],["-5:00","-","EST","2001","Apr","1","3:00"],["-6:00","Canada","C%sT"]],"America/Cambridge_Bay":[["0","-","zzz","1920",""],["-7:00","NT_YK","M%sT","1999","Oct","31","2:00"],["-6:00","Canada","C%sT","2000","Oct","29","2:00"],["-5:00","-","EST","2000","Nov","5","0:00"],["-6:00","-","CST","2001","Apr","1","3:00"],["-7:00","Canada","M%sT"]],"America/Yellowknife":[["0","-","zzz","1935",""],["-7:00","NT_YK","M%sT","1980"],["-7:00","Canada","M%sT"]],"America/Inuvik":[["0","-","zzz","1953",""],["-8:00","NT_YK","P%sT","1979","Apr","lastSun","2:00"],["-7:00","NT_YK","M%sT","1980"],["-7:00","Canada","M%sT"]],"America/Whitehorse":[["-9:00:12","-","LMT","1900","Aug","20"],["-9:00","NT_YK","Y%sT","1966","Jul","1","2:00"],["-8:00","NT_YK","P%sT","1980"],["-8:00","Canada","P%sT"]],"America/Dawson":[["-9:17:40","-","LMT","1900","Aug","20"],["-9:00","NT_YK","Y%sT","1973","Oct","28","0:00"],["-8:00","NT_YK","P%sT","1980"],["-8:00","Canada","P%sT"]],"America/Cancun":[["-5:47:04","-","LMT","1922","Jan","1","0:12:56"],["-6:00","-","CST","1981","Dec","23"],["-5:00","Mexico","E%sT","1998","Aug","2","2:00"],["-6:00","Mexico","C%sT"]],"America/Merida":[["-5:58:28","-","LMT","1922","Jan","1","0:01:32"],["-6:00","-","CST","1981","Dec","23"],["-5:00","-","EST","1982","Dec","2"],["-6:00","Mexico","C%sT"]],"America/Matamoros":[["-6:40:00","-","LMT","1921","Dec","31","23:20:00"],["-6:00","-","CST","1988"],["-6:00","US","C%sT","1989"],["-6:00","Mexico","C%sT","2010"],["-6:00","US","C%sT"]],"America/Monterrey":[["-6:41:16","-","LMT","1921","Dec","31","23:18:44"],["-6:00","-","CST","1988"],["-6:00","US","C%sT","1989"],["-6:00","Mexico","C%sT"]],"America/Mexico_City":[["-6:36:36","-","LMT","1922","Jan","1","0:23:24"],["-7:00","-","MST","1927","Jun","10","23:00"],["-6:00","-","CST","1930","Nov","15"],["-7:00","-","MST","1931","May","1","23:00"],["-6:00","-","CST","1931","Oct"],["-7:00","-","MST","1932","Apr","1"],["-6:00","Mexico","C%sT","2001","Sep","30","02:00"],["-6:00","-","CST","2002","Feb","20"],["-6:00","Mexico","C%sT"]],"America/Ojinaga":[["-6:57:40","-","LMT","1922","Jan","1","0:02:20"],["-7:00","-","MST","1927","Jun","10","23:00"],["-6:00","-","CST","1930","Nov","15"],["-7:00","-","MST","1931","May","1","23:00"],["-6:00","-","CST","1931","Oct"],["-7:00","-","MST","1932","Apr","1"],["-6:00","-","CST","1996"],["-6:00","Mexico","C%sT","1998"],["-6:00","-","CST","1998","Apr","Sun>=1","3:00"],["-7:00","Mexico","M%sT","2010"],["-7:00","US","M%sT"]],"America/Chihuahua":[["-7:04:20","-","LMT","1921","Dec","31","23:55:40"],["-7:00","-","MST","1927","Jun","10","23:00"],["-6:00","-","CST","1930","Nov","15"],["-7:00","-","MST","1931","May","1","23:00"],["-6:00","-","CST","1931","Oct"],["-7:00","-","MST","1932","Apr","1"],["-6:00","-","CST","1996"],["-6:00","Mexico","C%sT","1998"],["-6:00","-","CST","1998","Apr","Sun>=1","3:00"],["-7:00","Mexico","M%sT"]],"America/Hermosillo":[["-7:23:52","-","LMT","1921","Dec","31","23:36:08"],["-7:00","-","MST","1927","Jun","10","23:00"],["-6:00","-","CST","1930","Nov","15"],["-7:00","-","MST","1931","May","1","23:00"],["-6:00","-","CST","1931","Oct"],["-7:00","-","MST","1932","Apr","1"],["-6:00","-","CST","1942","Apr","24"],["-7:00","-","MST","1949","Jan","14"],["-8:00","-","PST","1970"],["-7:00","Mexico","M%sT","1999"],["-7:00","-","MST"]],"America/Mazatlan":[["-7:05:40","-","LMT","1921","Dec","31","23:54:20"],["-7:00","-","MST","1927","Jun","10","23:00"],["-6:00","-","CST","1930","Nov","15"],["-7:00","-","MST","1931","May","1","23:00"],["-6:00","-","CST","1931","Oct"],["-7:00","-","MST","1932","Apr","1"],["-6:00","-","CST","1942","Apr","24"],["-7:00","-","MST","1949","Jan","14"],["-8:00","-","PST","1970"],["-7:00","Mexico","M%sT"]],"America/Bahia_Banderas":[["-7:01:00","-","LMT","1921","Dec","31","23:59:00"],["-7:00","-","MST","1927","Jun","10","23:00"],["-6:00","-","CST","1930","Nov","15"],["-7:00","-","MST","1931","May","1","23:00"],["-6:00","-","CST","1931","Oct"],["-7:00","-","MST","1932","Apr","1"],["-6:00","-","CST","1942","Apr","24"],["-7:00","-","MST","1949","Jan","14"],["-8:00","-","PST","1970"],["-7:00","Mexico","M%sT","2010","Apr","4","2:00"],["-6:00","Mexico","C%sT"]],"America/Tijuana":[["-7:48:04","-","LMT","1922","Jan","1","0:11:56"],["-7:00","-","MST","1924"],["-8:00","-","PST","1927","Jun","10","23:00"],["-7:00","-","MST","1930","Nov","15"],["-8:00","-","PST","1931","Apr","1"],["-8:00","1:00","PDT","1931","Sep","30"],["-8:00","-","PST","1942","Apr","24"],["-8:00","1:00","PWT","1945","Aug","14","23:00u"],["-8:00","1:00","PPT","1945","Nov","12",""],["-8:00","-","PST","1948","Apr","5"],["-8:00","1:00","PDT","1949","Jan","14"],["-8:00","-","PST","1954"],["-8:00","CA","P%sT","1961"],["-8:00","-","PST","1976"],["-8:00","US","P%sT","1996"],["-8:00","Mexico","P%sT","2001"],["-8:00","US","P%sT","2002","Feb","20"],["-8:00","Mexico","P%sT","2010"],["-8:00","US","P%sT"]],"America/Santa_Isabel":[["-7:39:28","-","LMT","1922","Jan","1","0:20:32"],["-7:00","-","MST","1924"],["-8:00","-","PST","1927","Jun","10","23:00"],["-7:00","-","MST","1930","Nov","15"],["-8:00","-","PST","1931","Apr","1"],["-8:00","1:00","PDT","1931","Sep","30"],["-8:00","-","PST","1942","Apr","24"],["-8:00","1:00","PWT","1945","Aug","14","23:00u"],["-8:00","1:00","PPT","1945","Nov","12",""],["-8:00","-","PST","1948","Apr","5"],["-8:00","1:00","PDT","1949","Jan","14"],["-8:00","-","PST","1954"],["-8:00","CA","P%sT","1961"],["-8:00","-","PST","1976"],["-8:00","US","P%sT","1996"],["-8:00","Mexico","P%sT","2001"],["-8:00","US","P%sT","2002","Feb","20"],["-8:00","Mexico","P%sT"]],"America/Anguilla":[["-4:12:16","-","LMT","1912","Mar","2"],["-4:00","-","AST"]],"America/Antigua":[["-4:07:12","-","LMT","1912","Mar","2"],["-5:00","-","EST","1951"],["-4:00","-","AST"]],"America/Nassau":[["-5:09:24","-","LMT","1912","Mar","2"],["-5:00","Bahamas","E%sT","1976"],["-5:00","US","E%sT"]],"America/Barbados":[["-3:58:28","-","LMT","1924",""],["-3:58:28","-","BMT","1932",""],["-4:00","Barb","A%sT"]],"America/Belize":[["-5:52:48","-","LMT","1912","Apr"],["-6:00","Belize","C%sT"]],"Atlantic/Bermuda":[["-4:19:04","-","LMT","1930","Jan","1","2:00",""],["-4:00","-","AST","1974","Apr","28","2:00"],["-4:00","Bahamas","A%sT","1976"],["-4:00","US","A%sT"]],"America/Cayman":[["-5:25:32","-","LMT","1890",""],["-5:07:12","-","KMT","1912","Feb",""],["-5:00","-","EST"]],"America/Costa_Rica":[["-5:36:20","-","LMT","1890",""],["-5:36:20","-","SJMT","1921","Jan","15",""],["-6:00","CR","C%sT"]],"America/Havana":[["-5:29:28","-","LMT","1890"],["-5:29:36","-","HMT","1925","Jul","19","12:00",""],["-5:00","Cuba","C%sT"]],"America/Dominica":[["-4:05:36","-","LMT","1911","Jul","1","0:01",""],["-4:00","-","AST"]],"America/Santo_Domingo":[["-4:39:36","-","LMT","1890"],["-4:40","-","SDMT","1933","Apr","1","12:00",""],["-5:00","DR","E%sT","1974","Oct","27"],["-4:00","-","AST","2000","Oct","29","02:00"],["-5:00","US","E%sT","2000","Dec","3","01:00"],["-4:00","-","AST"]],"America/El_Salvador":[["-5:56:48","-","LMT","1921",""],["-6:00","Salv","C%sT"]],"America/Grenada":[["-4:07:00","-","LMT","1911","Jul",""],["-4:00","-","AST"]],"America/Guadeloupe":[["-4:06:08","-","LMT","1911","Jun","8",""],["-4:00","-","AST"]],"America/St_Barthelemy":"America/Guadeloupe","America/Marigot":"America/Guadeloupe","America/Guatemala":[["-6:02:04","-","LMT","1918","Oct","5"],["-6:00","Guat","C%sT"]],"America/Port-au-Prince":[["-4:49:20","-","LMT","1890"],["-4:49","-","PPMT","1917","Jan","24","12:00",""],["-5:00","Haiti","E%sT"]],"America/Tegucigalpa":[["-5:48:52","-","LMT","1921","Apr"],["-6:00","Hond","C%sT"]],"America/Jamaica":[["-5:07:12","-","LMT","1890",""],["-5:07:12","-","KMT","1912","Feb",""],["-5:00","-","EST","1974","Apr","28","2:00"],["-5:00","US","E%sT","1984"],["-5:00","-","EST"]],"America/Martinique":[["-4:04:20","-","LMT","1890",""],["-4:04:20","-","FFMT","1911","May",""],["-4:00","-","AST","1980","Apr","6"],["-4:00","1:00","ADT","1980","Sep","28"],["-4:00","-","AST"]],"America/Montserrat":[["-4:08:52","-","LMT","1911","Jul","1","0:01",""],["-4:00","-","AST"]],"America/Managua":[["-5:45:08","-","LMT","1890"],["-5:45:12","-","MMT","1934","Jun","23",""],["-6:00","-","CST","1973","May"],["-5:00","-","EST","1975","Feb","16"],["-6:00","Nic","C%sT","1992","Jan","1","4:00"],["-5:00","-","EST","1992","Sep","24"],["-6:00","-","CST","1993"],["-5:00","-","EST","1997"],["-6:00","Nic","C%sT"]],"America/Panama":[["-5:18:08","-","LMT","1890"],["-5:19:36","-","CMT","1908","Apr","22",""],["-5:00","-","EST"]],"America/Puerto_Rico":[["-4:24:25","-","LMT","1899","Mar","28","12:00",""],["-4:00","-","AST","1942","May","3"],["-4:00","US","A%sT","1946"],["-4:00","-","AST"]],"America/St_Kitts":[["-4:10:52","-","LMT","1912","Mar","2",""],["-4:00","-","AST"]],"America/St_Lucia":[["-4:04:00","-","LMT","1890",""],["-4:04:00","-","CMT","1912",""],["-4:00","-","AST"]],"America/Miquelon":[["-3:44:40","-","LMT","1911","May","15",""],["-4:00","-","AST","1980","May"],["-3:00","-","PMST","1987",""],["-3:00","Canada","PM%sT"]],"America/St_Vincent":[["-4:04:56","-","LMT","1890",""],["-4:04:56","-","KMT","1912",""],["-4:00","-","AST"]],"America/Grand_Turk":[["-4:44:32","-","LMT","1890"],["-5:07:12","-","KMT","1912","Feb",""],["-5:00","TC","E%sT"]],"America/Tortola":[["-4:18:28","-","LMT","1911","Jul",""],["-4:00","-","AST"]],"America/St_Thomas":[["-4:19:44","-","LMT","1911","Jul",""],["-4:00","-","AST"]]};
timezoneJS.timezone.rules = {"GB-Eire":[["1916","only","-","May","21","2:00s","1:00","BST"],["1916","only","-","Oct","1","2:00s","0","GMT"],["1917","only","-","Apr","8","2:00s","1:00","BST"],["1917","only","-","Sep","17","2:00s","0","GMT"],["1918","only","-","Mar","24","2:00s","1:00","BST"],["1918","only","-","Sep","30","2:00s","0","GMT"],["1919","only","-","Mar","30","2:00s","1:00","BST"],["1919","only","-","Sep","29","2:00s","0","GMT"],["1920","only","-","Mar","28","2:00s","1:00","BST"],["1920","only","-","Oct","25","2:00s","0","GMT"],["1921","only","-","Apr","3","2:00s","1:00","BST"],["1921","only","-","Oct","3","2:00s","0","GMT"],["1922","only","-","Mar","26","2:00s","1:00","BST"],["1922","only","-","Oct","8","2:00s","0","GMT"],["1923","only","-","Apr","Sun>=16","2:00s","1:00","BST"],["1923","1924","-","Sep","Sun>=16","2:00s","0","GMT"],["1924","only","-","Apr","Sun>=9","2:00s","1:00","BST"],["1925","1926","-","Apr","Sun>=16","2:00s","1:00","BST"],["1925","1938","-","Oct","Sun>=2","2:00s","0","GMT"],["1927","only","-","Apr","Sun>=9","2:00s","1:00","BST"],["1928","1929","-","Apr","Sun>=16","2:00s","1:00","BST"],["1930","only","-","Apr","Sun>=9","2:00s","1:00","BST"],["1931","1932","-","Apr","Sun>=16","2:00s","1:00","BST"],["1933","only","-","Apr","Sun>=9","2:00s","1:00","BST"],["1934","only","-","Apr","Sun>=16","2:00s","1:00","BST"],["1935","only","-","Apr","Sun>=9","2:00s","1:00","BST"],["1936","1937","-","Apr","Sun>=16","2:00s","1:00","BST"],["1938","only","-","Apr","Sun>=9","2:00s","1:00","BST"],["1939","only","-","Apr","Sun>=16","2:00s","1:00","BST"],["1939","only","-","Nov","Sun>=16","2:00s","0","GMT"],["1940","only","-","Feb","Sun>=23","2:00s","1:00","BST"],["1941","only","-","May","Sun>=2","1:00s","2:00","BDST"],["1941","1943","-","Aug","Sun>=9","1:00s","1:00","BST"],["1942","1944","-","Apr","Sun>=2","1:00s","2:00","BDST"],["1944","only","-","Sep","Sun>=16","1:00s","1:00","BST"],["1945","only","-","Apr","Mon>=2","1:00s","2:00","BDST"],["1945","only","-","Jul","Sun>=9","1:00s","1:00","BST"],["1945","1946","-","Oct","Sun>=2","2:00s","0","GMT"],["1946","only","-","Apr","Sun>=9","2:00s","1:00","BST"],["1947","only","-","Mar","16","2:00s","1:00","BST"],["1947","only","-","Apr","13","1:00s","2:00","BDST"],["1947","only","-","Aug","10","1:00s","1:00","BST"],["1947","only","-","Nov","2","2:00s","0","GMT"],["1948","only","-","Mar","14","2:00s","1:00","BST"],["1948","only","-","Oct","31","2:00s","0","GMT"],["1949","only","-","Apr","3","2:00s","1:00","BST"],["1949","only","-","Oct","30","2:00s","0","GMT"],["1950","1952","-","Apr","Sun>=14","2:00s","1:00","BST"],["1950","1952","-","Oct","Sun>=21","2:00s","0","GMT"],["1953","only","-","Apr","Sun>=16","2:00s","1:00","BST"],["1953","1960","-","Oct","Sun>=2","2:00s","0","GMT"],["1954","only","-","Apr","Sun>=9","2:00s","1:00","BST"],["1955","1956","-","Apr","Sun>=16","2:00s","1:00","BST"],["1957","only","-","Apr","Sun>=9","2:00s","1:00","BST"],["1958","1959","-","Apr","Sun>=16","2:00s","1:00","BST"],["1960","only","-","Apr","Sun>=9","2:00s","1:00","BST"],["1961","1963","-","Mar","lastSun","2:00s","1:00","BST"],["1961","1968","-","Oct","Sun>=23","2:00s","0","GMT"],["1964","1967","-","Mar","Sun>=19","2:00s","1:00","BST"],["1968","only","-","Feb","18","2:00s","1:00","BST"],["1972","1980","-","Mar","Sun>=16","2:00s","1:00","BST"],["1972","1980","-","Oct","Sun>=23","2:00s","0","GMT"],["1981","1995","-","Mar","lastSun","1:00u","1:00","BST"],["1981","1989","-","Oct","Sun>=23","1:00u","0","GMT"],["1990","1995","-","Oct","Sun>=22","1:00u","0","GMT"]],"EU":[["1977","1980","-","Apr","Sun>=1","1:00u","1:00","S"],["1977","only","-","Sep","lastSun","1:00u","0","-"],["1978","only","-","Oct","1","1:00u","0","-"],["1979","1995","-","Sep","lastSun","1:00u","0","-"],["1981","max","-","Mar","lastSun","1:00u","1:00","S"],["1996","max","-","Oct","lastSun","1:00u","0","-"]],"W-Eur":[["1977","1980","-","Apr","Sun>=1","1:00s","1:00","S"],["1977","only","-","Sep","lastSun","1:00s","0","-"],["1978","only","-","Oct","1","1:00s","0","-"],["1979","1995","-","Sep","lastSun","1:00s","0","-"],["1981","max","-","Mar","lastSun","1:00s","1:00","S"],["1996","max","-","Oct","lastSun","1:00s","0","-"]],"C-Eur":[["1916","only","-","Apr","30","23:00","1:00","S"],["1916","only","-","Oct","1","1:00","0","-"],["1917","1918","-","Apr","Mon>=15","2:00s","1:00","S"],["1917","1918","-","Sep","Mon>=15","2:00s","0","-"],["1940","only","-","Apr","1","2:00s","1:00","S"],["1942","only","-","Nov","2","2:00s","0","-"],["1943","only","-","Mar","29","2:00s","1:00","S"],["1943","only","-","Oct","4","2:00s","0","-"],["1944","1945","-","Apr","Mon>=1","2:00s","1:00","S"],["1944","only","-","Oct","2","2:00s","0","-"],["1945","only","-","Sep","16","2:00s","0","-"],["1977","1980","-","Apr","Sun>=1","2:00s","1:00","S"],["1977","only","-","Sep","lastSun","2:00s","0","-"],["1978","only","-","Oct","1","2:00s","0","-"],["1979","1995","-","Sep","lastSun","2:00s","0","-"],["1981","max","-","Mar","lastSun","2:00s","1:00","S"],["1996","max","-","Oct","lastSun","2:00s","0","-"]],"E-Eur":[["1977","1980","-","Apr","Sun>=1","0:00","1:00","S"],["1977","only","-","Sep","lastSun","0:00","0","-"],["1978","only","-","Oct","1","0:00","0","-"],["1979","1995","-","Sep","lastSun","0:00","0","-"],["1981","max","-","Mar","lastSun","0:00","1:00","S"],["1996","max","-","Oct","lastSun","0:00","0","-"]],"Russia":[["1917","only","-","Jul","1","23:00","1:00","MST",""],["1917","only","-","Dec","28","0:00","0","MMT",""],["1918","only","-","May","31","22:00","2:00","MDST",""],["1918","only","-","Sep","16","1:00","1:00","MST"],["1919","only","-","May","31","23:00","2:00","MDST"],["1919","only","-","Jul","1","2:00","1:00","S"],["1919","only","-","Aug","16","0:00","0","-"],["1921","only","-","Feb","14","23:00","1:00","S"],["1921","only","-","Mar","20","23:00","2:00","M",""],["1921","only","-","Sep","1","0:00","1:00","S"],["1921","only","-","Oct","1","0:00","0","-"],["1981","1984","-","Apr","1","0:00","1:00","S"],["1981","1983","-","Oct","1","0:00","0","-"],["1984","1991","-","Sep","lastSun","2:00s","0","-"],["1985","1991","-","Mar","lastSun","2:00s","1:00","S"],["1992","only","-","Mar","lastSat","23:00","1:00","S"],["1992","only","-","Sep","lastSat","23:00","0","-"],["1993","max","-","Mar","lastSun","2:00s","1:00","S"],["1993","1995","-","Sep","lastSun","2:00s","0","-"],["1996","max","-","Oct","lastSun","2:00s","0","-"]],"Albania":[["1940","only","-","Jun","16","0:00","1:00","S"],["1942","only","-","Nov","2","3:00","0","-"],["1943","only","-","Mar","29","2:00","1:00","S"],["1943","only","-","Apr","10","3:00","0","-"],["1974","only","-","May","4","0:00","1:00","S"],["1974","only","-","Oct","2","0:00","0","-"],["1975","only","-","May","1","0:00","1:00","S"],["1975","only","-","Oct","2","0:00","0","-"],["1976","only","-","May","2","0:00","1:00","S"],["1976","only","-","Oct","3","0:00","0","-"],["1977","only","-","May","8","0:00","1:00","S"],["1977","only","-","Oct","2","0:00","0","-"],["1978","only","-","May","6","0:00","1:00","S"],["1978","only","-","Oct","1","0:00","0","-"],["1979","only","-","May","5","0:00","1:00","S"],["1979","only","-","Sep","30","0:00","0","-"],["1980","only","-","May","3","0:00","1:00","S"],["1980","only","-","Oct","4","0:00","0","-"],["1981","only","-","Apr","26","0:00","1:00","S"],["1981","only","-","Sep","27","0:00","0","-"],["1982","only","-","May","2","0:00","1:00","S"],["1982","only","-","Oct","3","0:00","0","-"],["1983","only","-","Apr","18","0:00","1:00","S"],["1983","only","-","Oct","1","0:00","0","-"],["1984","only","-","Apr","1","0:00","1:00","S"]],"Austria":[["1920","only","-","Apr","5","2:00s","1:00","S"],["1920","only","-","Sep","13","2:00s","0","-"],["1946","only","-","Apr","14","2:00s","1:00","S"],["1946","1948","-","Oct","Sun>=1","2:00s","0","-"],["1947","only","-","Apr","6","2:00s","1:00","S"],["1948","only","-","Apr","18","2:00s","1:00","S"],["1980","only","-","Apr","6","0:00","1:00","S"],["1980","only","-","Sep","28","0:00","0","-"]],"Belgium":[["1918","only","-","Mar","9","0:00s","1:00","S"],["1918","1919","-","Oct","Sat>=1","23:00s","0","-"],["1919","only","-","Mar","1","23:00s","1:00","S"],["1920","only","-","Feb","14","23:00s","1:00","S"],["1920","only","-","Oct","23","23:00s","0","-"],["1921","only","-","Mar","14","23:00s","1:00","S"],["1921","only","-","Oct","25","23:00s","0","-"],["1922","only","-","Mar","25","23:00s","1:00","S"],["1922","1927","-","Oct","Sat>=1","23:00s","0","-"],["1923","only","-","Apr","21","23:00s","1:00","S"],["1924","only","-","Mar","29","23:00s","1:00","S"],["1925","only","-","Apr","4","23:00s","1:00","S"],["1926","only","-","Apr","17","23:00s","1:00","S"],["1927","only","-","Apr","9","23:00s","1:00","S"],["1928","only","-","Apr","14","23:00s","1:00","S"],["1928","1938","-","Oct","Sun>=2","2:00s","0","-"],["1929","only","-","Apr","21","2:00s","1:00","S"],["1930","only","-","Apr","13","2:00s","1:00","S"],["1931","only","-","Apr","19","2:00s","1:00","S"],["1932","only","-","Apr","3","2:00s","1:00","S"],["1933","only","-","Mar","26","2:00s","1:00","S"],["1934","only","-","Apr","8","2:00s","1:00","S"],["1935","only","-","Mar","31","2:00s","1:00","S"],["1936","only","-","Apr","19","2:00s","1:00","S"],["1937","only","-","Apr","4","2:00s","1:00","S"],["1938","only","-","Mar","27","2:00s","1:00","S"],["1939","only","-","Apr","16","2:00s","1:00","S"],["1939","only","-","Nov","19","2:00s","0","-"],["1940","only","-","Feb","25","2:00s","1:00","S"],["1944","only","-","Sep","17","2:00s","0","-"],["1945","only","-","Apr","2","2:00s","1:00","S"],["1945","only","-","Sep","16","2:00s","0","-"],["1946","only","-","May","19","2:00s","1:00","S"],["1946","only","-","Oct","7","2:00s","0","-"]],"Bulg":[["1979","only","-","Mar","31","23:00","1:00","S"],["1979","only","-","Oct","1","1:00","0","-"],["1980","1982","-","Apr","Sat>=1","23:00","1:00","S"],["1980","only","-","Sep","29","1:00","0","-"],["1981","only","-","Sep","27","2:00","0","-"]],"Czech":[["1945","only","-","Apr","8","2:00s","1:00","S"],["1945","only","-","Nov","18","2:00s","0","-"],["1946","only","-","May","6","2:00s","1:00","S"],["1946","1949","-","Oct","Sun>=1","2:00s","0","-"],["1947","only","-","Apr","20","2:00s","1:00","S"],["1948","only","-","Apr","18","2:00s","1:00","S"],["1949","only","-","Apr","9","2:00s","1:00","S"]],"Denmark":[["1916","only","-","May","14","23:00","1:00","S"],["1916","only","-","Sep","30","23:00","0","-"],["1940","only","-","May","15","0:00","1:00","S"],["1945","only","-","Apr","2","2:00s","1:00","S"],["1945","only","-","Aug","15","2:00s","0","-"],["1946","only","-","May","1","2:00s","1:00","S"],["1946","only","-","Sep","1","2:00s","0","-"],["1947","only","-","May","4","2:00s","1:00","S"],["1947","only","-","Aug","10","2:00s","0","-"],["1948","only","-","May","9","2:00s","1:00","S"],["1948","only","-","Aug","8","2:00s","0","-"]],"Thule":[["1991","1992","-","Mar","lastSun","2:00","1:00","D"],["1991","1992","-","Sep","lastSun","2:00","0","S"],["1993","2006","-","Apr","Sun>=1","2:00","1:00","D"],["1993","2006","-","Oct","lastSun","2:00","0","S"],["2007","max","-","Mar","Sun>=8","2:00","1:00","D"],["2007","max","-","Nov","Sun>=1","2:00","0","S"]],"Finland":[["1942","only","-","Apr","3","0:00","1:00","S"],["1942","only","-","Oct","3","0:00","0","-"],["1981","1982","-","Mar","lastSun","2:00","1:00","S"],["1981","1982","-","Sep","lastSun","3:00","0","-"]],"France":[["1916","only","-","Jun","14","23:00s","1:00","S"],["1916","1919","-","Oct","Sun>=1","23:00s","0","-"],["1917","only","-","Mar","24","23:00s","1:00","S"],["1918","only","-","Mar","9","23:00s","1:00","S"],["1919","only","-","Mar","1","23:00s","1:00","S"],["1920","only","-","Feb","14","23:00s","1:00","S"],["1920","only","-","Oct","23","23:00s","0","-"],["1921","only","-","Mar","14","23:00s","1:00","S"],["1921","only","-","Oct","25","23:00s","0","-"],["1922","only","-","Mar","25","23:00s","1:00","S"],["1922","1938","-","Oct","Sat>=1","23:00s","0","-"],["1923","only","-","May","26","23:00s","1:00","S"],["1924","only","-","Mar","29","23:00s","1:00","S"],["1925","only","-","Apr","4","23:00s","1:00","S"],["1926","only","-","Apr","17","23:00s","1:00","S"],["1927","only","-","Apr","9","23:00s","1:00","S"],["1928","only","-","Apr","14","23:00s","1:00","S"],["1929","only","-","Apr","20","23:00s","1:00","S"],["1930","only","-","Apr","12","23:00s","1:00","S"],["1931","only","-","Apr","18","23:00s","1:00","S"],["1932","only","-","Apr","2","23:00s","1:00","S"],["1933","only","-","Mar","25","23:00s","1:00","S"],["1934","only","-","Apr","7","23:00s","1:00","S"],["1935","only","-","Mar","30","23:00s","1:00","S"],["1936","only","-","Apr","18","23:00s","1:00","S"],["1937","only","-","Apr","3","23:00s","1:00","S"],["1938","only","-","Mar","26","23:00s","1:00","S"],["1939","only","-","Apr","15","23:00s","1:00","S"],["1939","only","-","Nov","18","23:00s","0","-"],["1940","only","-","Feb","25","2:00","1:00","S"],["1941","only","-","May","5","0:00","2:00","M",""],["1941","only","-","Oct","6","0:00","1:00","S"],["1942","only","-","Mar","9","0:00","2:00","M"],["1942","only","-","Nov","2","3:00","1:00","S"],["1943","only","-","Mar","29","2:00","2:00","M"],["1943","only","-","Oct","4","3:00","1:00","S"],["1944","only","-","Apr","3","2:00","2:00","M"],["1944","only","-","Oct","8","1:00","1:00","S"],["1945","only","-","Apr","2","2:00","2:00","M"],["1945","only","-","Sep","16","3:00","0","-"],["1976","only","-","Mar","28","1:00","1:00","S"],["1976","only","-","Sep","26","1:00","0","-"]],"Germany":[["1946","only","-","Apr","14","2:00s","1:00","S"],["1946","only","-","Oct","7","2:00s","0","-"],["1947","1949","-","Oct","Sun>=1","2:00s","0","-"],["1947","only","-","Apr","6","3:00s","1:00","S"],["1947","only","-","May","11","2:00s","2:00","M"],["1947","only","-","Jun","29","3:00","1:00","S"],["1948","only","-","Apr","18","2:00s","1:00","S"],["1949","only","-","Apr","10","2:00s","1:00","S"]],"SovietZone":[["1945","only","-","May","24","2:00","2:00","M",""],["1945","only","-","Sep","24","3:00","1:00","S"],["1945","only","-","Nov","18","2:00s","0","-"]],"Greece":[["1932","only","-","Jul","7","0:00","1:00","S"],["1932","only","-","Sep","1","0:00","0","-"],["1941","only","-","Apr","7","0:00","1:00","S"],["1942","only","-","Nov","2","3:00","0","-"],["1943","only","-","Mar","30","0:00","1:00","S"],["1943","only","-","Oct","4","0:00","0","-"],["1952","only","-","Jul","1","0:00","1:00","S"],["1952","only","-","Nov","2","0:00","0","-"],["1975","only","-","Apr","12","0:00s","1:00","S"],["1975","only","-","Nov","26","0:00s","0","-"],["1976","only","-","Apr","11","2:00s","1:00","S"],["1976","only","-","Oct","10","2:00s","0","-"],["1977","1978","-","Apr","Sun>=1","2:00s","1:00","S"],["1977","only","-","Sep","26","2:00s","0","-"],["1978","only","-","Sep","24","4:00","0","-"],["1979","only","-","Apr","1","9:00","1:00","S"],["1979","only","-","Sep","29","2:00","0","-"],["1980","only","-","Apr","1","0:00","1:00","S"],["1980","only","-","Sep","28","0:00","0","-"]],"Hungary":[["1918","only","-","Apr","1","3:00","1:00","S"],["1918","only","-","Sep","29","3:00","0","-"],["1919","only","-","Apr","15","3:00","1:00","S"],["1919","only","-","Sep","15","3:00","0","-"],["1920","only","-","Apr","5","3:00","1:00","S"],["1920","only","-","Sep","30","3:00","0","-"],["1945","only","-","May","1","23:00","1:00","S"],["1945","only","-","Nov","3","0:00","0","-"],["1946","only","-","Mar","31","2:00s","1:00","S"],["1946","1949","-","Oct","Sun>=1","2:00s","0","-"],["1947","1949","-","Apr","Sun>=4","2:00s","1:00","S"],["1950","only","-","Apr","17","2:00s","1:00","S"],["1950","only","-","Oct","23","2:00s","0","-"],["1954","1955","-","May","23","0:00","1:00","S"],["1954","1955","-","Oct","3","0:00","0","-"],["1956","only","-","Jun","Sun>=1","0:00","1:00","S"],["1956","only","-","Sep","lastSun","0:00","0","-"],["1957","only","-","Jun","Sun>=1","1:00","1:00","S"],["1957","only","-","Sep","lastSun","3:00","0","-"],["1980","only","-","Apr","6","1:00","1:00","S"]],"Iceland":[["1917","1918","-","Feb","19","23:00","1:00","S"],["1917","only","-","Oct","21","1:00","0","-"],["1918","only","-","Nov","16","1:00","0","-"],["1939","only","-","Apr","29","23:00","1:00","S"],["1939","only","-","Nov","29","2:00","0","-"],["1940","only","-","Feb","25","2:00","1:00","S"],["1940","only","-","Nov","3","2:00","0","-"],["1941","only","-","Mar","2","1:00s","1:00","S"],["1941","only","-","Nov","2","1:00s","0","-"],["1942","only","-","Mar","8","1:00s","1:00","S"],["1942","only","-","Oct","25","1:00s","0","-"],["1943","1946","-","Mar","Sun>=1","1:00s","1:00","S"],["1943","1948","-","Oct","Sun>=22","1:00s","0","-"],["1947","1967","-","Apr","Sun>=1","1:00s","1:00","S"],["1949","only","-","Oct","30","1:00s","0","-"],["1950","1966","-","Oct","Sun>=22","1:00s","0","-"],["1967","only","-","Oct","29","1:00s","0","-"]],"Italy":[["1916","only","-","Jun","3","0:00s","1:00","S"],["1916","only","-","Oct","1","0:00s","0","-"],["1917","only","-","Apr","1","0:00s","1:00","S"],["1917","only","-","Sep","30","0:00s","0","-"],["1918","only","-","Mar","10","0:00s","1:00","S"],["1918","1919","-","Oct","Sun>=1","0:00s","0","-"],["1919","only","-","Mar","2","0:00s","1:00","S"],["1920","only","-","Mar","21","0:00s","1:00","S"],["1920","only","-","Sep","19","0:00s","0","-"],["1940","only","-","Jun","15","0:00s","1:00","S"],["1944","only","-","Sep","17","0:00s","0","-"],["1945","only","-","Apr","2","2:00","1:00","S"],["1945","only","-","Sep","15","0:00s","0","-"],["1946","only","-","Mar","17","2:00s","1:00","S"],["1946","only","-","Oct","6","2:00s","0","-"],["1947","only","-","Mar","16","0:00s","1:00","S"],["1947","only","-","Oct","5","0:00s","0","-"],["1948","only","-","Feb","29","2:00s","1:00","S"],["1948","only","-","Oct","3","2:00s","0","-"],["1966","1968","-","May","Sun>=22","0:00","1:00","S"],["1966","1969","-","Sep","Sun>=22","0:00","0","-"],["1969","only","-","Jun","1","0:00","1:00","S"],["1970","only","-","May","31","0:00","1:00","S"],["1970","only","-","Sep","lastSun","0:00","0","-"],["1971","1972","-","May","Sun>=22","0:00","1:00","S"],["1971","only","-","Sep","lastSun","1:00","0","-"],["1972","only","-","Oct","1","0:00","0","-"],["1973","only","-","Jun","3","0:00","1:00","S"],["1973","1974","-","Sep","lastSun","0:00","0","-"],["1974","only","-","May","26","0:00","1:00","S"],["1975","only","-","Jun","1","0:00s","1:00","S"],["1975","1977","-","Sep","lastSun","0:00s","0","-"],["1976","only","-","May","30","0:00s","1:00","S"],["1977","1979","-","May","Sun>=22","0:00s","1:00","S"],["1978","only","-","Oct","1","0:00s","0","-"],["1979","only","-","Sep","30","0:00s","0","-"]],"Latvia":[["1989","1996","-","Mar","lastSun","2:00s","1:00","S"],["1989","1996","-","Sep","lastSun","2:00s","0","-"]],"Lux":[["1916","only","-","May","14","23:00","1:00","S"],["1916","only","-","Oct","1","1:00","0","-"],["1917","only","-","Apr","28","23:00","1:00","S"],["1917","only","-","Sep","17","1:00","0","-"],["1918","only","-","Apr","Mon>=15","2:00s","1:00","S"],["1918","only","-","Sep","Mon>=15","2:00s","0","-"],["1919","only","-","Mar","1","23:00","1:00","S"],["1919","only","-","Oct","5","3:00","0","-"],["1920","only","-","Feb","14","23:00","1:00","S"],["1920","only","-","Oct","24","2:00","0","-"],["1921","only","-","Mar","14","23:00","1:00","S"],["1921","only","-","Oct","26","2:00","0","-"],["1922","only","-","Mar","25","23:00","1:00","S"],["1922","only","-","Oct","Sun>=2","1:00","0","-"],["1923","only","-","Apr","21","23:00","1:00","S"],["1923","only","-","Oct","Sun>=2","2:00","0","-"],["1924","only","-","Mar","29","23:00","1:00","S"],["1924","1928","-","Oct","Sun>=2","1:00","0","-"],["1925","only","-","Apr","5","23:00","1:00","S"],["1926","only","-","Apr","17","23:00","1:00","S"],["1927","only","-","Apr","9","23:00","1:00","S"],["1928","only","-","Apr","14","23:00","1:00","S"],["1929","only","-","Apr","20","23:00","1:00","S"]],"Malta":[["1973","only","-","Mar","31","0:00s","1:00","S"],["1973","only","-","Sep","29","0:00s","0","-"],["1974","only","-","Apr","21","0:00s","1:00","S"],["1974","only","-","Sep","16","0:00s","0","-"],["1975","1979","-","Apr","Sun>=15","2:00","1:00","S"],["1975","1980","-","Sep","Sun>=15","2:00","0","-"],["1980","only","-","Mar","31","2:00","1:00","S"]],"Neth":[["1916","only","-","May","1","0:00","1:00","NST",""],["1916","only","-","Oct","1","0:00","0","AMT",""],["1917","only","-","Apr","16","2:00s","1:00","NST"],["1917","only","-","Sep","17","2:00s","0","AMT"],["1918","1921","-","Apr","Mon>=1","2:00s","1:00","NST"],["1918","1921","-","Sep","lastMon","2:00s","0","AMT"],["1922","only","-","Mar","lastSun","2:00s","1:00","NST"],["1922","1936","-","Oct","Sun>=2","2:00s","0","AMT"],["1923","only","-","Jun","Fri>=1","2:00s","1:00","NST"],["1924","only","-","Mar","lastSun","2:00s","1:00","NST"],["1925","only","-","Jun","Fri>=1","2:00s","1:00","NST"],["1926","1931","-","May","15","2:00s","1:00","NST"],["1932","only","-","May","22","2:00s","1:00","NST"],["1933","1936","-","May","15","2:00s","1:00","NST"],["1937","only","-","May","22","2:00s","1:00","NST"],["1937","only","-","Jul","1","0:00","1:00","S"],["1937","1939","-","Oct","Sun>=2","2:00s","0","-"],["1938","1939","-","May","15","2:00s","1:00","S"],["1945","only","-","Apr","2","2:00s","1:00","S"],["1945","only","-","Sep","16","2:00s","0","-"]],"Norway":[["1916","only","-","May","22","1:00","1:00","S"],["1916","only","-","Sep","30","0:00","0","-"],["1945","only","-","Apr","2","2:00s","1:00","S"],["1945","only","-","Oct","1","2:00s","0","-"],["1959","1964","-","Mar","Sun>=15","2:00s","1:00","S"],["1959","1965","-","Sep","Sun>=15","2:00s","0","-"],["1965","only","-","Apr","25","2:00s","1:00","S"]],"Poland":[["1918","1919","-","Sep","16","2:00s","0","-"],["1919","only","-","Apr","15","2:00s","1:00","S"],["1944","only","-","Apr","3","2:00s","1:00","S"],["1944","only","-","Oct","4","2:00","0","-"],["1945","only","-","Apr","29","0:00","1:00","S"],["1945","only","-","Nov","1","0:00","0","-"],["1946","only","-","Apr","14","0:00s","1:00","S"],["1946","only","-","Oct","7","2:00s","0","-"],["1947","only","-","May","4","2:00s","1:00","S"],["1947","1949","-","Oct","Sun>=1","2:00s","0","-"],["1948","only","-","Apr","18","2:00s","1:00","S"],["1949","only","-","Apr","10","2:00s","1:00","S"],["1957","only","-","Jun","2","1:00s","1:00","S"],["1957","1958","-","Sep","lastSun","1:00s","0","-"],["1958","only","-","Mar","30","1:00s","1:00","S"],["1959","only","-","May","31","1:00s","1:00","S"],["1959","1961","-","Oct","Sun>=1","1:00s","0","-"],["1960","only","-","Apr","3","1:00s","1:00","S"],["1961","1964","-","May","lastSun","1:00s","1:00","S"],["1962","1964","-","Sep","lastSun","1:00s","0","-"]],"Port":[["1916","only","-","Jun","17","23:00","1:00","S"],["1916","only","-","Nov","1","1:00","0","-"],["1917","only","-","Feb","28","23:00s","1:00","S"],["1917","1921","-","Oct","14","23:00s","0","-"],["1918","only","-","Mar","1","23:00s","1:00","S"],["1919","only","-","Feb","28","23:00s","1:00","S"],["1920","only","-","Feb","29","23:00s","1:00","S"],["1921","only","-","Feb","28","23:00s","1:00","S"],["1924","only","-","Apr","16","23:00s","1:00","S"],["1924","only","-","Oct","14","23:00s","0","-"],["1926","only","-","Apr","17","23:00s","1:00","S"],["1926","1929","-","Oct","Sat>=1","23:00s","0","-"],["1927","only","-","Apr","9","23:00s","1:00","S"],["1928","only","-","Apr","14","23:00s","1:00","S"],["1929","only","-","Apr","20","23:00s","1:00","S"],["1931","only","-","Apr","18","23:00s","1:00","S"],["1931","1932","-","Oct","Sat>=1","23:00s","0","-"],["1932","only","-","Apr","2","23:00s","1:00","S"],["1934","only","-","Apr","7","23:00s","1:00","S"],["1934","1938","-","Oct","Sat>=1","23:00s","0","-"],["1935","only","-","Mar","30","23:00s","1:00","S"],["1936","only","-","Apr","18","23:00s","1:00","S"],["1937","only","-","Apr","3","23:00s","1:00","S"],["1938","only","-","Mar","26","23:00s","1:00","S"],["1939","only","-","Apr","15","23:00s","1:00","S"],["1939","only","-","Nov","18","23:00s","0","-"],["1940","only","-","Feb","24","23:00s","1:00","S"],["1940","1941","-","Oct","5","23:00s","0","-"],["1941","only","-","Apr","5","23:00s","1:00","S"],["1942","1945","-","Mar","Sat>=8","23:00s","1:00","S"],["1942","only","-","Apr","25","22:00s","2:00","M",""],["1942","only","-","Aug","15","22:00s","1:00","S"],["1942","1945","-","Oct","Sat>=24","23:00s","0","-"],["1943","only","-","Apr","17","22:00s","2:00","M"],["1943","1945","-","Aug","Sat>=25","22:00s","1:00","S"],["1944","1945","-","Apr","Sat>=21","22:00s","2:00","M"],["1946","only","-","Apr","Sat>=1","23:00s","1:00","S"],["1946","only","-","Oct","Sat>=1","23:00s","0","-"],["1947","1949","-","Apr","Sun>=1","2:00s","1:00","S"],["1947","1949","-","Oct","Sun>=1","2:00s","0","-"],["1951","1965","-","Apr","Sun>=1","2:00s","1:00","S"],["1951","1965","-","Oct","Sun>=1","2:00s","0","-"],["1977","only","-","Mar","27","0:00s","1:00","S"],["1977","only","-","Sep","25","0:00s","0","-"],["1978","1979","-","Apr","Sun>=1","0:00s","1:00","S"],["1978","only","-","Oct","1","0:00s","0","-"],["1979","1982","-","Sep","lastSun","1:00s","0","-"],["1980","only","-","Mar","lastSun","0:00s","1:00","S"],["1981","1982","-","Mar","lastSun","1:00s","1:00","S"],["1983","only","-","Mar","lastSun","2:00s","1:00","S"]],"Romania":[["1932","only","-","May","21","0:00s","1:00","S"],["1932","1939","-","Oct","Sun>=1","0:00s","0","-"],["1933","1939","-","Apr","Sun>=2","0:00s","1:00","S"],["1979","only","-","May","27","0:00","1:00","S"],["1979","only","-","Sep","lastSun","0:00","0","-"],["1980","only","-","Apr","5","23:00","1:00","S"],["1980","only","-","Sep","lastSun","1:00","0","-"],["1991","1993","-","Mar","lastSun","0:00s","1:00","S"],["1991","1993","-","Sep","lastSun","0:00s","0","-"]],"Spain":[["1917","only","-","May","5","23:00s","1:00","S"],["1917","1919","-","Oct","6","23:00s","0","-"],["1918","only","-","Apr","15","23:00s","1:00","S"],["1919","only","-","Apr","5","23:00s","1:00","S"],["1924","only","-","Apr","16","23:00s","1:00","S"],["1924","only","-","Oct","4","23:00s","0","-"],["1926","only","-","Apr","17","23:00s","1:00","S"],["1926","1929","-","Oct","Sat>=1","23:00s","0","-"],["1927","only","-","Apr","9","23:00s","1:00","S"],["1928","only","-","Apr","14","23:00s","1:00","S"],["1929","only","-","Apr","20","23:00s","1:00","S"],["1937","only","-","May","22","23:00s","1:00","S"],["1937","1939","-","Oct","Sat>=1","23:00s","0","-"],["1938","only","-","Mar","22","23:00s","1:00","S"],["1939","only","-","Apr","15","23:00s","1:00","S"],["1940","only","-","Mar","16","23:00s","1:00","S"],["1942","only","-","May","2","22:00s","2:00","M",""],["1942","only","-","Sep","1","22:00s","1:00","S"],["1943","1946","-","Apr","Sat>=13","22:00s","2:00","M"],["1943","only","-","Oct","3","22:00s","1:00","S"],["1944","only","-","Oct","10","22:00s","1:00","S"],["1945","only","-","Sep","30","1:00","1:00","S"],["1946","only","-","Sep","30","0:00","0","-"],["1949","only","-","Apr","30","23:00","1:00","S"],["1949","only","-","Sep","30","1:00","0","-"],["1974","1975","-","Apr","Sat>=13","23:00","1:00","S"],["1974","1975","-","Oct","Sun>=1","1:00","0","-"],["1976","only","-","Mar","27","23:00","1:00","S"],["1976","1977","-","Sep","lastSun","1:00","0","-"],["1977","1978","-","Apr","2","23:00","1:00","S"],["1978","only","-","Oct","1","1:00","0","-"]],"SpainAfrica":[["1967","only","-","Jun","3","12:00","1:00","S"],["1967","only","-","Oct","1","0:00","0","-"],["1974","only","-","Jun","24","0:00","1:00","S"],["1974","only","-","Sep","1","0:00","0","-"],["1976","1977","-","May","1","0:00","1:00","S"],["1976","only","-","Aug","1","0:00","0","-"],["1977","only","-","Sep","28","0:00","0","-"],["1978","only","-","Jun","1","0:00","1:00","S"],["1978","only","-","Aug","4","0:00","0","-"]],"Swiss":[["1941","1942","-","May","Mon>=1","1:00","1:00","S"],["1941","1942","-","Oct","Mon>=1","2:00","0","-"]],"Turkey":[["1916","only","-","May","1","0:00","1:00","S"],["1916","only","-","Oct","1","0:00","0","-"],["1920","only","-","Mar","28","0:00","1:00","S"],["1920","only","-","Oct","25","0:00","0","-"],["1921","only","-","Apr","3","0:00","1:00","S"],["1921","only","-","Oct","3","0:00","0","-"],["1922","only","-","Mar","26","0:00","1:00","S"],["1922","only","-","Oct","8","0:00","0","-"],["1924","only","-","May","13","0:00","1:00","S"],["1924","1925","-","Oct","1","0:00","0","-"],["1925","only","-","May","1","0:00","1:00","S"],["1940","only","-","Jun","30","0:00","1:00","S"],["1940","only","-","Oct","5","0:00","0","-"],["1940","only","-","Dec","1","0:00","1:00","S"],["1941","only","-","Sep","21","0:00","0","-"],["1942","only","-","Apr","1","0:00","1:00","S"],["1942","only","-","Nov","1","0:00","0","-"],["1945","only","-","Apr","2","0:00","1:00","S"],["1945","only","-","Oct","8","0:00","0","-"],["1946","only","-","Jun","1","0:00","1:00","S"],["1946","only","-","Oct","1","0:00","0","-"],["1947","1948","-","Apr","Sun>=16","0:00","1:00","S"],["1947","1950","-","Oct","Sun>=2","0:00","0","-"],["1949","only","-","Apr","10","0:00","1:00","S"],["1950","only","-","Apr","19","0:00","1:00","S"],["1951","only","-","Apr","22","0:00","1:00","S"],["1951","only","-","Oct","8","0:00","0","-"],["1962","only","-","Jul","15","0:00","1:00","S"],["1962","only","-","Oct","8","0:00","0","-"],["1964","only","-","May","15","0:00","1:00","S"],["1964","only","-","Oct","1","0:00","0","-"],["1970","1972","-","May","Sun>=2","0:00","1:00","S"],["1970","1972","-","Oct","Sun>=2","0:00","0","-"],["1973","only","-","Jun","3","1:00","1:00","S"],["1973","only","-","Nov","4","3:00","0","-"],["1974","only","-","Mar","31","2:00","1:00","S"],["1974","only","-","Nov","3","5:00","0","-"],["1975","only","-","Mar","30","0:00","1:00","S"],["1975","1976","-","Oct","lastSun","0:00","0","-"],["1976","only","-","Jun","1","0:00","1:00","S"],["1977","1978","-","Apr","Sun>=1","0:00","1:00","S"],["1977","only","-","Oct","16","0:00","0","-"],["1979","1980","-","Apr","Sun>=1","3:00","1:00","S"],["1979","1982","-","Oct","Mon>=11","0:00","0","-"],["1981","1982","-","Mar","lastSun","3:00","1:00","S"],["1983","only","-","Jul","31","0:00","1:00","S"],["1983","only","-","Oct","2","0:00","0","-"],["1985","only","-","Apr","20","0:00","1:00","S"],["1985","only","-","Sep","28","0:00","0","-"],["1986","1990","-","Mar","lastSun","2:00s","1:00","S"],["1986","1990","-","Sep","lastSun","2:00s","0","-"],["1991","2006","-","Mar","lastSun","1:00s","1:00","S"],["1991","1995","-","Sep","lastSun","1:00s","0","-"],["1996","2006","-","Oct","lastSun","1:00s","0","-"]],"US":[["1918","1919","-","Mar","lastSun","2:00","1:00","D"],["1918","1919","-","Oct","lastSun","2:00","0","S"],["1942","only","-","Feb","9","2:00","1:00","W",""],["1945","only","-","Aug","14","23:00u","1:00","P",""],["1945","only","-","Sep","30","2:00","0","S"],["1967","2006","-","Oct","lastSun","2:00","0","S"],["1967","1973","-","Apr","lastSun","2:00","1:00","D"],["1974","only","-","Jan","6","2:00","1:00","D"],["1975","only","-","Feb","23","2:00","1:00","D"],["1976","1986","-","Apr","lastSun","2:00","1:00","D"],["1987","2006","-","Apr","Sun>=1","2:00","1:00","D"],["2007","max","-","Mar","Sun>=8","2:00","1:00","D"],["2007","max","-","Nov","Sun>=1","2:00","0","S"]],"NYC":[["1920","only","-","Mar","lastSun","2:00","1:00","D"],["1920","only","-","Oct","lastSun","2:00","0","S"],["1921","1966","-","Apr","lastSun","2:00","1:00","D"],["1921","1954","-","Sep","lastSun","2:00","0","S"],["1955","1966","-","Oct","lastSun","2:00","0","S"]],"Chicago":[["1920","only","-","Jun","13","2:00","1:00","D"],["1920","1921","-","Oct","lastSun","2:00","0","S"],["1921","only","-","Mar","lastSun","2:00","1:00","D"],["1922","1966","-","Apr","lastSun","2:00","1:00","D"],["1922","1954","-","Sep","lastSun","2:00","0","S"],["1955","1966","-","Oct","lastSun","2:00","0","S"]],"Denver":[["1920","1921","-","Mar","lastSun","2:00","1:00","D"],["1920","only","-","Oct","lastSun","2:00","0","S"],["1921","only","-","May","22","2:00","0","S"],["1965","1966","-","Apr","lastSun","2:00","1:00","D"],["1965","1966","-","Oct","lastSun","2:00","0","S"]],"CA":[["1948","only","-","Mar","14","2:00","1:00","D"],["1949","only","-","Jan","1","2:00","0","S"],["1950","1966","-","Apr","lastSun","2:00","1:00","D"],["1950","1961","-","Sep","lastSun","2:00","0","S"],["1962","1966","-","Oct","lastSun","2:00","0","S"]],"Indianapolis":[["1941","only","-","Jun","22","2:00","1:00","D"],["1941","1954","-","Sep","lastSun","2:00","0","S"],["1946","1954","-","Apr","lastSun","2:00","1:00","D"]],"Marengo":[["1951","only","-","Apr","lastSun","2:00","1:00","D"],["1951","only","-","Sep","lastSun","2:00","0","S"],["1954","1960","-","Apr","lastSun","2:00","1:00","D"],["1954","1960","-","Sep","lastSun","2:00","0","S"]],"Vincennes":[["1946","only","-","Apr","lastSun","2:00","1:00","D"],["1946","only","-","Sep","lastSun","2:00","0","S"],["1953","1954","-","Apr","lastSun","2:00","1:00","D"],["1953","1959","-","Sep","lastSun","2:00","0","S"],["1955","only","-","May","1","0:00","1:00","D"],["1956","1963","-","Apr","lastSun","2:00","1:00","D"],["1960","only","-","Oct","lastSun","2:00","0","S"],["1961","only","-","Sep","lastSun","2:00","0","S"],["1962","1963","-","Oct","lastSun","2:00","0","S"]],"Perry":[["1946","only","-","Apr","lastSun","2:00","1:00","D"],["1946","only","-","Sep","lastSun","2:00","0","S"],["1953","1954","-","Apr","lastSun","2:00","1:00","D"],["1953","1959","-","Sep","lastSun","2:00","0","S"],["1955","only","-","May","1","0:00","1:00","D"],["1956","1963","-","Apr","lastSun","2:00","1:00","D"],["1960","only","-","Oct","lastSun","2:00","0","S"],["1961","only","-","Sep","lastSun","2:00","0","S"],["1962","1963","-","Oct","lastSun","2:00","0","S"]],"Pike":[["1955","only","-","May","1","0:00","1:00","D"],["1955","1960","-","Sep","lastSun","2:00","0","S"],["1956","1964","-","Apr","lastSun","2:00","1:00","D"],["1961","1964","-","Oct","lastSun","2:00","0","S"]],"Starke":[["1947","1961","-","Apr","lastSun","2:00","1:00","D"],["1947","1954","-","Sep","lastSun","2:00","0","S"],["1955","1956","-","Oct","lastSun","2:00","0","S"],["1957","1958","-","Sep","lastSun","2:00","0","S"],["1959","1961","-","Oct","lastSun","2:00","0","S"]],"Pulaski":[["1946","1960","-","Apr","lastSun","2:00","1:00","D"],["1946","1954","-","Sep","lastSun","2:00","0","S"],["1955","1956","-","Oct","lastSun","2:00","0","S"],["1957","1960","-","Sep","lastSun","2:00","0","S"]],"Louisville":[["1921","only","-","May","1","2:00","1:00","D"],["1921","only","-","Sep","1","2:00","0","S"],["1941","1961","-","Apr","lastSun","2:00","1:00","D"],["1941","only","-","Sep","lastSun","2:00","0","S"],["1946","only","-","Jun","2","2:00","0","S"],["1950","1955","-","Sep","lastSun","2:00","0","S"],["1956","1960","-","Oct","lastSun","2:00","0","S"]],"Detroit":[["1948","only","-","Apr","lastSun","2:00","1:00","D"],["1948","only","-","Sep","lastSun","2:00","0","S"],["1967","only","-","Jun","14","2:00","1:00","D"],["1967","only","-","Oct","lastSun","2:00","0","S"]],"Menominee":[["1946","only","-","Apr","lastSun","2:00","1:00","D"],["1946","only","-","Sep","lastSun","2:00","0","S"],["1966","only","-","Apr","lastSun","2:00","1:00","D"],["1966","only","-","Oct","lastSun","2:00","0","S"]],"Canada":[["1918","only","-","Apr","14","2:00","1:00","D"],["1918","only","-","Oct","31","2:00","0","S"],["1942","only","-","Feb","9","2:00","1:00","W",""],["1945","only","-","Aug","14","23:00u","1:00","P",""],["1945","only","-","Sep","30","2:00","0","S"],["1974","1986","-","Apr","lastSun","2:00","1:00","D"],["1974","2006","-","Oct","lastSun","2:00","0","S"],["1987","2006","-","Apr","Sun>=1","2:00","1:00","D"],["2007","max","-","Mar","Sun>=8","2:00","1:00","D"],["2007","max","-","Nov","Sun>=1","2:00","0","S"]],"StJohns":[["1917","only","-","Apr","8","2:00","1:00","D"],["1917","only","-","Sep","17","2:00","0","S"],["1919","only","-","May","5","23:00","1:00","D"],["1919","only","-","Aug","12","23:00","0","S"],["1920","1935","-","May","Sun>=1","23:00","1:00","D"],["1920","1935","-","Oct","lastSun","23:00","0","S"],["1936","1941","-","May","Mon>=9","0:00","1:00","D"],["1936","1941","-","Oct","Mon>=2","0:00","0","S"],["1946","1950","-","May","Sun>=8","2:00","1:00","D"],["1946","1950","-","Oct","Sun>=2","2:00","0","S"],["1951","1986","-","Apr","lastSun","2:00","1:00","D"],["1951","1959","-","Sep","lastSun","2:00","0","S"],["1960","1986","-","Oct","lastSun","2:00","0","S"],["1987","only","-","Apr","Sun>=1","0:01","1:00","D"],["1987","2006","-","Oct","lastSun","0:01","0","S"],["1988","only","-","Apr","Sun>=1","0:01","2:00","DD"],["1989","2006","-","Apr","Sun>=1","0:01","1:00","D"],["2007","max","-","Mar","Sun>=8","0:01","1:00","D"],["2007","max","-","Nov","Sun>=1","0:01","0","S"]],"Halifax":[["1916","only","-","Apr","1","0:00","1:00","D"],["1916","only","-","Oct","1","0:00","0","S"],["1920","only","-","May","9","0:00","1:00","D"],["1920","only","-","Aug","29","0:00","0","S"],["1921","only","-","May","6","0:00","1:00","D"],["1921","1922","-","Sep","5","0:00","0","S"],["1922","only","-","Apr","30","0:00","1:00","D"],["1923","1925","-","May","Sun>=1","0:00","1:00","D"],["1923","only","-","Sep","4","0:00","0","S"],["1924","only","-","Sep","15","0:00","0","S"],["1925","only","-","Sep","28","0:00","0","S"],["1926","only","-","May","16","0:00","1:00","D"],["1926","only","-","Sep","13","0:00","0","S"],["1927","only","-","May","1","0:00","1:00","D"],["1927","only","-","Sep","26","0:00","0","S"],["1928","1931","-","May","Sun>=8","0:00","1:00","D"],["1928","only","-","Sep","9","0:00","0","S"],["1929","only","-","Sep","3","0:00","0","S"],["1930","only","-","Sep","15","0:00","0","S"],["1931","1932","-","Sep","Mon>=24","0:00","0","S"],["1932","only","-","May","1","0:00","1:00","D"],["1933","only","-","Apr","30","0:00","1:00","D"],["1933","only","-","Oct","2","0:00","0","S"],["1934","only","-","May","20","0:00","1:00","D"],["1934","only","-","Sep","16","0:00","0","S"],["1935","only","-","Jun","2","0:00","1:00","D"],["1935","only","-","Sep","30","0:00","0","S"],["1936","only","-","Jun","1","0:00","1:00","D"],["1936","only","-","Sep","14","0:00","0","S"],["1937","1938","-","May","Sun>=1","0:00","1:00","D"],["1937","1941","-","Sep","Mon>=24","0:00","0","S"],["1939","only","-","May","28","0:00","1:00","D"],["1940","1941","-","May","Sun>=1","0:00","1:00","D"],["1946","1949","-","Apr","lastSun","2:00","1:00","D"],["1946","1949","-","Sep","lastSun","2:00","0","S"],["1951","1954","-","Apr","lastSun","2:00","1:00","D"],["1951","1954","-","Sep","lastSun","2:00","0","S"],["1956","1959","-","Apr","lastSun","2:00","1:00","D"],["1956","1959","-","Sep","lastSun","2:00","0","S"],["1962","1973","-","Apr","lastSun","2:00","1:00","D"],["1962","1973","-","Oct","lastSun","2:00","0","S"]],"Moncton":[["1933","1935","-","Jun","Sun>=8","1:00","1:00","D"],["1933","1935","-","Sep","Sun>=8","1:00","0","S"],["1936","1938","-","Jun","Sun>=1","1:00","1:00","D"],["1936","1938","-","Sep","Sun>=1","1:00","0","S"],["1939","only","-","May","27","1:00","1:00","D"],["1939","1941","-","Sep","Sat>=21","1:00","0","S"],["1940","only","-","May","19","1:00","1:00","D"],["1941","only","-","May","4","1:00","1:00","D"],["1946","1972","-","Apr","lastSun","2:00","1:00","D"],["1946","1956","-","Sep","lastSun","2:00","0","S"],["1957","1972","-","Oct","lastSun","2:00","0","S"],["1993","2006","-","Apr","Sun>=1","0:01","1:00","D"],["1993","2006","-","Oct","lastSun","0:01","0","S"]],"Mont":[["1917","only","-","Mar","25","2:00","1:00","D"],["1917","only","-","Apr","24","0:00","0","S"],["1919","only","-","Mar","31","2:30","1:00","D"],["1919","only","-","Oct","25","2:30","0","S"],["1920","only","-","May","2","2:30","1:00","D"],["1920","1922","-","Oct","Sun>=1","2:30","0","S"],["1921","only","-","May","1","2:00","1:00","D"],["1922","only","-","Apr","30","2:00","1:00","D"],["1924","only","-","May","17","2:00","1:00","D"],["1924","1926","-","Sep","lastSun","2:30","0","S"],["1925","1926","-","May","Sun>=1","2:00","1:00","D"],["1927","only","-","May","1","0:00","1:00","D"],["1927","1932","-","Sep","lastSun","0:00","0","S"],["1928","1931","-","Apr","lastSun","0:00","1:00","D"],["1932","only","-","May","1","0:00","1:00","D"],["1933","1940","-","Apr","lastSun","0:00","1:00","D"],["1933","only","-","Oct","1","0:00","0","S"],["1934","1939","-","Sep","lastSun","0:00","0","S"],["1946","1973","-","Apr","lastSun","2:00","1:00","D"],["1945","1948","-","Sep","lastSun","2:00","0","S"],["1949","1950","-","Oct","lastSun","2:00","0","S"],["1951","1956","-","Sep","lastSun","2:00","0","S"],["1957","1973","-","Oct","lastSun","2:00","0","S"]],"Toronto":[["1919","only","-","Mar","30","23:30","1:00","D"],["1919","only","-","Oct","26","0:00","0","S"],["1920","only","-","May","2","2:00","1:00","D"],["1920","only","-","Sep","26","0:00","0","S"],["1921","only","-","May","15","2:00","1:00","D"],["1921","only","-","Sep","15","2:00","0","S"],["1922","1923","-","May","Sun>=8","2:00","1:00","D"],["1922","1926","-","Sep","Sun>=15","2:00","0","S"],["1924","1927","-","May","Sun>=1","2:00","1:00","D"],["1927","1932","-","Sep","lastSun","2:00","0","S"],["1928","1931","-","Apr","lastSun","2:00","1:00","D"],["1932","only","-","May","1","2:00","1:00","D"],["1933","1940","-","Apr","lastSun","2:00","1:00","D"],["1933","only","-","Oct","1","2:00","0","S"],["1934","1939","-","Sep","lastSun","2:00","0","S"],["1945","1946","-","Sep","lastSun","2:00","0","S"],["1946","only","-","Apr","lastSun","2:00","1:00","D"],["1947","1949","-","Apr","lastSun","0:00","1:00","D"],["1947","1948","-","Sep","lastSun","0:00","0","S"],["1949","only","-","Nov","lastSun","0:00","0","S"],["1950","1973","-","Apr","lastSun","2:00","1:00","D"],["1950","only","-","Nov","lastSun","2:00","0","S"],["1951","1956","-","Sep","lastSun","2:00","0","S"],["1957","1973","-","Oct","lastSun","2:00","0","S"]],"Winn":[["1916","only","-","Apr","23","0:00","1:00","D"],["1916","only","-","Sep","17","0:00","0","S"],["1918","only","-","Apr","14","2:00","1:00","D"],["1918","only","-","Oct","31","2:00","0","S"],["1937","only","-","May","16","2:00","1:00","D"],["1937","only","-","Sep","26","2:00","0","S"],["1942","only","-","Feb","9","2:00","1:00","W",""],["1945","only","-","Aug","14","23:00u","1:00","P",""],["1945","only","-","Sep","lastSun","2:00","0","S"],["1946","only","-","May","12","2:00","1:00","D"],["1946","only","-","Oct","13","2:00","0","S"],["1947","1949","-","Apr","lastSun","2:00","1:00","D"],["1947","1949","-","Sep","lastSun","2:00","0","S"],["1950","only","-","May","1","2:00","1:00","D"],["1950","only","-","Sep","30","2:00","0","S"],["1951","1960","-","Apr","lastSun","2:00","1:00","D"],["1951","1958","-","Sep","lastSun","2:00","0","S"],["1959","only","-","Oct","lastSun","2:00","0","S"],["1960","only","-","Sep","lastSun","2:00","0","S"],["1963","only","-","Apr","lastSun","2:00","1:00","D"],["1963","only","-","Sep","22","2:00","0","S"],["1966","1986","-","Apr","lastSun","2:00s","1:00","D"],["1966","2005","-","Oct","lastSun","2:00s","0","S"],["1987","2005","-","Apr","Sun>=1","2:00s","1:00","D"]],"Regina":[["1918","only","-","Apr","14","2:00","1:00","D"],["1918","only","-","Oct","31","2:00","0","S"],["1930","1934","-","May","Sun>=1","0:00","1:00","D"],["1930","1934","-","Oct","Sun>=1","0:00","0","S"],["1937","1941","-","Apr","Sun>=8","0:00","1:00","D"],["1937","only","-","Oct","Sun>=8","0:00","0","S"],["1938","only","-","Oct","Sun>=1","0:00","0","S"],["1939","1941","-","Oct","Sun>=8","0:00","0","S"],["1942","only","-","Feb","9","2:00","1:00","W",""],["1945","only","-","Aug","14","23:00u","1:00","P",""],["1945","only","-","Sep","lastSun","2:00","0","S"],["1946","only","-","Apr","Sun>=8","2:00","1:00","D"],["1946","only","-","Oct","Sun>=8","2:00","0","S"],["1947","1957","-","Apr","lastSun","2:00","1:00","D"],["1947","1957","-","Sep","lastSun","2:00","0","S"],["1959","only","-","Apr","lastSun","2:00","1:00","D"],["1959","only","-","Oct","lastSun","2:00","0","S"]],"Swift":[["1957","only","-","Apr","lastSun","2:00","1:00","D"],["1957","only","-","Oct","lastSun","2:00","0","S"],["1959","1961","-","Apr","lastSun","2:00","1:00","D"],["1959","only","-","Oct","lastSun","2:00","0","S"],["1960","1961","-","Sep","lastSun","2:00","0","S"]],"Edm":[["1918","1919","-","Apr","Sun>=8","2:00","1:00","D"],["1918","only","-","Oct","31","2:00","0","S"],["1919","only","-","May","27","2:00","0","S"],["1920","1923","-","Apr","lastSun","2:00","1:00","D"],["1920","only","-","Oct","lastSun","2:00","0","S"],["1921","1923","-","Sep","lastSun","2:00","0","S"],["1942","only","-","Feb","9","2:00","1:00","W",""],["1945","only","-","Aug","14","23:00u","1:00","P",""],["1945","only","-","Sep","lastSun","2:00","0","S"],["1947","only","-","Apr","lastSun","2:00","1:00","D"],["1947","only","-","Sep","lastSun","2:00","0","S"],["1967","only","-","Apr","lastSun","2:00","1:00","D"],["1967","only","-","Oct","lastSun","2:00","0","S"],["1969","only","-","Apr","lastSun","2:00","1:00","D"],["1969","only","-","Oct","lastSun","2:00","0","S"],["1972","1986","-","Apr","lastSun","2:00","1:00","D"],["1972","2006","-","Oct","lastSun","2:00","0","S"]],"Vanc":[["1918","only","-","Apr","14","2:00","1:00","D"],["1918","only","-","Oct","31","2:00","0","S"],["1942","only","-","Feb","9","2:00","1:00","W",""],["1945","only","-","Aug","14","23:00u","1:00","P",""],["1945","only","-","Sep","30","2:00","0","S"],["1946","1986","-","Apr","lastSun","2:00","1:00","D"],["1946","only","-","Oct","13","2:00","0","S"],["1947","1961","-","Sep","lastSun","2:00","0","S"],["1962","2006","-","Oct","lastSun","2:00","0","S"]],"NT_YK":[["1918","only","-","Apr","14","2:00","1:00","D"],["1918","only","-","Oct","27","2:00","0","S"],["1919","only","-","May","25","2:00","1:00","D"],["1919","only","-","Nov","1","0:00","0","S"],["1942","only","-","Feb","9","2:00","1:00","W",""],["1945","only","-","Aug","14","23:00u","1:00","P",""],["1945","only","-","Sep","30","2:00","0","S"],["1965","only","-","Apr","lastSun","0:00","2:00","DD"],["1965","only","-","Oct","lastSun","2:00","0","S"],["1980","1986","-","Apr","lastSun","2:00","1:00","D"],["1980","2006","-","Oct","lastSun","2:00","0","S"],["1987","2006","-","Apr","Sun>=1","2:00","1:00","D"]],"Resolute":[["2006","max","-","Nov","Sun>=1","2:00","0","ES"],["2007","max","-","Mar","Sun>=8","2:00","0","CD"]],"Mexico":[["1939","only","-","Feb","5","0:00","1:00","D"],["1939","only","-","Jun","25","0:00","0","S"],["1940","only","-","Dec","9","0:00","1:00","D"],["1941","only","-","Apr","1","0:00","0","S"],["1943","only","-","Dec","16","0:00","1:00","W",""],["1944","only","-","May","1","0:00","0","S"],["1950","only","-","Feb","12","0:00","1:00","D"],["1950","only","-","Jul","30","0:00","0","S"],["1996","2000","-","Apr","Sun>=1","2:00","1:00","D"],["1996","2000","-","Oct","lastSun","2:00","0","S"],["2001","only","-","May","Sun>=1","2:00","1:00","D"],["2001","only","-","Sep","lastSun","2:00","0","S"],["2002","max","-","Apr","Sun>=1","2:00","1:00","D"],["2002","max","-","Oct","lastSun","2:00","0","S"]],"Bahamas":[["1964","1975","-","Oct","lastSun","2:00","0","S"],["1964","1975","-","Apr","lastSun","2:00","1:00","D"]],"Barb":[["1977","only","-","Jun","12","2:00","1:00","D"],["1977","1978","-","Oct","Sun>=1","2:00","0","S"],["1978","1980","-","Apr","Sun>=15","2:00","1:00","D"],["1979","only","-","Sep","30","2:00","0","S"],["1980","only","-","Sep","25","2:00","0","S"]],"Belize":[["1918","1942","-","Oct","Sun>=2","0:00","0:30","HD"],["1919","1943","-","Feb","Sun>=9","0:00","0","S"],["1973","only","-","Dec","5","0:00","1:00","D"],["1974","only","-","Feb","9","0:00","0","S"],["1982","only","-","Dec","18","0:00","1:00","D"],["1983","only","-","Feb","12","0:00","0","S"]],"CR":[["1979","1980","-","Feb","lastSun","0:00","1:00","D"],["1979","1980","-","Jun","Sun>=1","0:00","0","S"],["1991","1992","-","Jan","Sat>=15","0:00","1:00","D"],["1991","only","-","Jul","1","0:00","0","S"],["1992","only","-","Mar","15","0:00","0","S"]],"Cuba":[["1928","only","-","Jun","10","0:00","1:00","D"],["1928","only","-","Oct","10","0:00","0","S"],["1940","1942","-","Jun","Sun>=1","0:00","1:00","D"],["1940","1942","-","Sep","Sun>=1","0:00","0","S"],["1945","1946","-","Jun","Sun>=1","0:00","1:00","D"],["1945","1946","-","Sep","Sun>=1","0:00","0","S"],["1965","only","-","Jun","1","0:00","1:00","D"],["1965","only","-","Sep","30","0:00","0","S"],["1966","only","-","May","29","0:00","1:00","D"],["1966","only","-","Oct","2","0:00","0","S"],["1967","only","-","Apr","8","0:00","1:00","D"],["1967","1968","-","Sep","Sun>=8","0:00","0","S"],["1968","only","-","Apr","14","0:00","1:00","D"],["1969","1977","-","Apr","lastSun","0:00","1:00","D"],["1969","1971","-","Oct","lastSun","0:00","0","S"],["1972","1974","-","Oct","8","0:00","0","S"],["1975","1977","-","Oct","lastSun","0:00","0","S"],["1978","only","-","May","7","0:00","1:00","D"],["1978","1990","-","Oct","Sun>=8","0:00","0","S"],["1979","1980","-","Mar","Sun>=15","0:00","1:00","D"],["1981","1985","-","May","Sun>=5","0:00","1:00","D"],["1986","1989","-","Mar","Sun>=14","0:00","1:00","D"],["1990","1997","-","Apr","Sun>=1","0:00","1:00","D"],["1991","1995","-","Oct","Sun>=8","0:00s","0","S"],["1996","only","-","Oct","6","0:00s","0","S"],["1997","only","-","Oct","12","0:00s","0","S"],["1998","1999","-","Mar","lastSun","0:00s","1:00","D"],["1998","2003","-","Oct","lastSun","0:00s","0","S"],["2000","2004","-","Apr","Sun>=1","0:00s","1:00","D"],["2006","max","-","Oct","lastSun","0:00s","0","S"],["2007","only","-","Mar","Sun>=8","0:00s","1:00","D"],["2008","only","-","Mar","Sun>=15","0:00s","1:00","D"],["2009","2010","-","Mar","Sun>=8","0:00s","1:00","D"],["2011","only","-","Mar","Sun>=15","0:00s","1:00","D"],["2012","max","-","Mar","Sun>=8","0:00s","1:00","D"]],"DR":[["1966","only","-","Oct","30","0:00","1:00","D"],["1967","only","-","Feb","28","0:00","0","S"],["1969","1973","-","Oct","lastSun","0:00","0:30","HD"],["1970","only","-","Feb","21","0:00","0","S"],["1971","only","-","Jan","20","0:00","0","S"],["1972","1974","-","Jan","21","0:00","0","S"]],"Salv":[["1987","1988","-","May","Sun>=1","0:00","1:00","D"],["1987","1988","-","Sep","lastSun","0:00","0","S"]],"Guat":[["1973","only","-","Nov","25","0:00","1:00","D"],["1974","only","-","Feb","24","0:00","0","S"],["1983","only","-","May","21","0:00","1:00","D"],["1983","only","-","Sep","22","0:00","0","S"],["1991","only","-","Mar","23","0:00","1:00","D"],["1991","only","-","Sep","7","0:00","0","S"],["2006","only","-","Apr","30","0:00","1:00","D"],["2006","only","-","Oct","1","0:00","0","S"]],"Haiti":[["1983","only","-","May","8","0:00","1:00","D"],["1984","1987","-","Apr","lastSun","0:00","1:00","D"],["1983","1987","-","Oct","lastSun","0:00","0","S"],["1988","1997","-","Apr","Sun>=1","1:00s","1:00","D"],["1988","1997","-","Oct","lastSun","1:00s","0","S"],["2005","2006","-","Apr","Sun>=1","0:00","1:00","D"],["2005","2006","-","Oct","lastSun","0:00","0","S"]],"Hond":[["1987","1988","-","May","Sun>=1","0:00","1:00","D"],["1987","1988","-","Sep","lastSun","0:00","0","S"],["2006","only","-","May","Sun>=1","0:00","1:00","D"],["2006","only","-","Aug","Mon>=1","0:00","0","S"]],"Nic":[["1979","1980","-","Mar","Sun>=16","0:00","1:00","D"],["1979","1980","-","Jun","Mon>=23","0:00","0","S"],["2005","only","-","Apr","10","0:00","1:00","D"],["2005","only","-","Oct","Sun>=1","0:00","0","S"],["2006","only","-","Apr","30","2:00","1:00","D"],["2006","only","-","Oct","Sun>=1","1:00","0","S"]],"TC":[["1979","1986","-","Apr","lastSun","2:00","1:00","D"],["1979","2006","-","Oct","lastSun","2:00","0","S"],["1987","2006","-","Apr","Sun>=1","2:00","1:00","D"],["2007","max","-","Mar","Sun>=8","2:00","1:00","D"],["2007","max","-","Nov","Sun>=1","2:00","0","S"]]};

/*
 * jQuery Impromptu
 * By: Trent Richardson [http://trentrichardson.com]
 * Version 3.1
 * Last Modified: 3/30/2010
 * 
 * Copyright 2010 Trent Richardson
 * Dual licensed under the MIT and GPL licenses.
 * http://trentrichardson.com/Impromptu/GPL-LICENSE.txt
 * http://trentrichardson.com/Impromptu/MIT-LICENSE.txt
 * 
 */
 
(function($) {
  $.prompt = function(message, options) {
    options = $.extend({},$.prompt.defaults,options);
    $.prompt.currentPrefix = options.prefix;

    var ie6   = ($.browser.msie && $.browser.version < 7);
    var $body = $(document.body);
    var $window = $(window);
    
    options.classes = $.trim(options.classes);
    if(options.classes != '')
      options.classes = ' '+ options.classes;
      
    //build the box and fade
    var msgbox = '<div class="'+ options.prefix +'box'+ options.classes +'" id="'+ options.prefix +'box">';
    if(options.useiframe && (($('object, applet').length > 0) || ie6)) {
      msgbox += '<iframe src="javascript:false;" style="display:block;position:absolute;z-index:-1;" class="'+ options.prefix +'fade" id="'+ options.prefix +'fade"></iframe>';
    } else {
      if(ie6) {
        $('select').css('visibility','hidden');
      }
      msgbox +='<div class="'+ options.prefix +'fade" id="'+ options.prefix +'fade"></div>';
    }
    msgbox += '<div class="'+ options.prefix +'" id="'+ options.prefix +'"><div class="'+ options.prefix +'container"><div class="';
    msgbox += options.prefix +'close">X</div><div id="'+ options.prefix +'states"></div>';
    msgbox += '</div></div></div>';

    var $jqib = $(msgbox).appendTo($body);
    var $jqi  = $jqib.children('#'+ options.prefix);
    var $jqif = $jqib.children('#'+ options.prefix +'fade');

    //if a string was passed, convert to a single state
    if(message.constructor == String){
      message = {
        state0: {
          html: message,
          buttons: options.buttons,
          focus: options.focus,
          submit: options.submit
        }
      };
    }

    //build the states
    var states = "";

    $.each(message,function(statename,stateobj){
      stateobj = $.extend({},$.prompt.defaults.state,stateobj);
      message[statename] = stateobj;

      states += '<div id="'+ options.prefix +'_state_'+ statename +'" class="'+ options.prefix + '_state" style="display:none;"><div class="'+ options.prefix +'message">' + stateobj.html +'</div><div class="'+ options.prefix +'buttons">';
      $.each(stateobj.buttons, function(k, v){
        if(typeof v == 'object')
          states += '<button name="' + options.prefix + '_' + statename + '_button' + v.title.replace(/[^a-z0-9]+/gi,'') + '" id="' + options.prefix + '_' + statename + '_button' + v.title.replace(/[^a-z0-9]+/gi,'') + '" value="' + v.value + '">' + v.title + '</button>';
        else states += '<button name="' + options.prefix + '_' + statename + '_button' + k + '" id="' + options.prefix +  '_' + statename + '_button' + k + '" value="' + v + '">' + k + '</button>';
      });
      states += '</div></div>';
    });

    //insert the states...
    $jqi.find('#'+ options.prefix +'states').html(states).children('.'+ options.prefix +'_state:first').css('display','block');
    $jqi.find('.'+ options.prefix +'buttons:empty').css('display','none');
    
    //Events
    $.each(message,function(statename,stateobj){
      var $state = $jqi.find('#'+ options.prefix +'_state_'+ statename);

      $state.children('.'+ options.prefix +'buttons').children('button').click(function(){
        var msg = $state.children('.'+ options.prefix +'message');
        var clicked = stateobj.buttons[$(this).text()];
        if(clicked == undefined){
          for(var i in stateobj.buttons)
            if(stateobj.buttons[i].title == $(this).text())
              clicked = stateobj.buttons[i].value;
        }
        
        if(typeof clicked == 'object')
          clicked = clicked.value;
        var forminputs = {};

        //collect all form element values from all states
        $.each($jqi.find('#'+ options.prefix +'states :input').serializeArray(),function(i,obj){
          if (forminputs[obj.name] === undefined) {
            forminputs[obj.name] = obj.value;
          } else if (typeof forminputs[obj.name] == Array || typeof forminputs[obj.name] == 'object') {
            forminputs[obj.name].push(obj.value);
          } else {
            forminputs[obj.name] = [forminputs[obj.name],obj.value];  
          } 
        });

        var close = stateobj.submit(clicked,msg,forminputs);
        if(close === undefined || close) {
          removePrompt(true,clicked,msg,forminputs);
        }
      });
      $state.find('.'+ options.prefix +'buttons button:eq('+ stateobj.focus +')').addClass(options.prefix +'defaultbutton');

    });

    var ie6scroll = function(){
      $jqib.css({ top: $window.scrollTop() });
    };

    var fadeClicked = function(){
      if(options.persistent){
        var i = 0;
        $jqib.addClass(options.prefix +'warning');
        var intervalid = setInterval(function(){
          $jqib.toggleClass(options.prefix +'warning');
          if(i++ > 1){
            clearInterval(intervalid);
            $jqib.removeClass(options.prefix +'warning');
          }
        }, 100);
      }
      else {
        removePrompt();
      }
    };
    
    var keyPressEventHandler = function(e){
      var key = (window.event) ? event.keyCode : e.keyCode; // MSIE or Firefox?
      
      //escape key closes
      if(key==27) {
        fadeClicked();  
      }
      
      //constrain tabs
      if (key == 9){
        var $inputels = $(':input:enabled:visible',$jqib);
        var fwd = !e.shiftKey && e.target == $inputels[$inputels.length-1];
        var back = e.shiftKey && e.target == $inputels[0];
        if (fwd || back) {
        setTimeout(function(){ 
          if (!$inputels)
            return;
          var el = $inputels[back===true ? $inputels.length-1 : 0];

          if (el)
            el.focus();           
        },10);
        return false;
        }
      }
    };
    
    var positionPrompt = function(){
      $jqib.css({
        position: (ie6) ? "absolute" : "fixed",
        height: $window.height(),
        width: "100%",
        top: (ie6)? $window.scrollTop() : 0,
        left: 0,
        right: 0,
        bottom: 0
      });
      $jqif.css({
        position: "absolute",
        height: $window.height(),
        width: "100%",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0
      });
      $jqi.css({
        position: "absolute",
        top: options.top,
        left: "50%",
        marginLeft: (($jqi.outerWidth()/2)*-1)
      });
    };

    var stylePrompt = function(){
      $jqif.css({
        zIndex: options.zIndex,
        display: "none",
        opacity: options.opacity
      });
      $jqi.css({
        zIndex: options.zIndex+1,
        display: "none"
      });
      $jqib.css({
        zIndex: options.zIndex
      });
    };

    var removePrompt = function(callCallback, clicked, msg, formvals){
      $jqi.remove();
      //ie6, remove the scroll event
      if(ie6) {
        $body.unbind('scroll',ie6scroll);
      }
      $window.unbind('resize',positionPrompt);
      $jqif.fadeOut(options.overlayspeed,function(){
        $jqif.unbind('click',fadeClicked);
        $jqif.remove();
        if(callCallback) {
          options.callback(clicked,msg,formvals);
        }
        $jqib.unbind('keypress',keyPressEventHandler);
        $jqib.remove();
        if(ie6 && !options.useiframe) {
          $('select').css('visibility','visible');
        }
      });
    };

    positionPrompt();
    stylePrompt();
    
    //ie6, add a scroll event to fix position:fixed
    if(ie6) {
      $window.scroll(ie6scroll);
    }
    $jqif.click(fadeClicked);
    $window.resize(positionPrompt);
    $jqib.bind("keydown keypress",keyPressEventHandler);
    $jqi.find('.'+ options.prefix +'close').click(removePrompt);

    //Show it
    $jqif.fadeIn(options.overlayspeed);
    $jqi[options.show](options.promptspeed,options.loaded);
    $jqi.find('#'+ options.prefix +'states .'+ options.prefix +'_state:first .'+ options.prefix +'defaultbutton').focus();
    
    if(options.timeout > 0)
      setTimeout($.prompt.close,options.timeout);

    return $jqib;
  };
  
  $.prompt.defaults = {
    prefix:'jqi',
    classes: '',
    buttons: {
      Ok: true
    },
    loaded: function(){

    },
      submit: function(){
        return true;
    },
    callback: function(){

    },
    opacity: 0.6,
    zIndex: 999,
      overlayspeed: 'slow',
      promptspeed: 'fast',
      show: 'fadeIn',
      focus: 0,
      useiframe: false,
    top: "15%",
      persistent: true,
      timeout: 0,
      state: {
      html: '',
      buttons: {
        Ok: true
      },
        focus: 0,
        submit: function(){
          return true;
       }
      }
  };
  
  $.prompt.currentPrefix = $.prompt.defaults.prefix;

  $.prompt.setDefaults = function(o) {
    $.prompt.defaults = $.extend({}, $.prompt.defaults, o);
  };
  
  $.prompt.setStateDefaults = function(o) {
    $.prompt.defaults.state = $.extend({}, $.prompt.defaults.state, o);
  };
  
  $.prompt.getStateContent = function(state) {
    return $('#'+ $.prompt.currentPrefix +'_state_'+ state);
  };
  
  $.prompt.getCurrentState = function() {
    return $('.'+ $.prompt.currentPrefix +'_state:visible');
  };
  
  $.prompt.getCurrentStateName = function() {
    var stateid = $.prompt.getCurrentState().attr('id');
    
    return stateid.replace($.prompt.currentPrefix +'_state_','');
  };
  
  $.prompt.goToState = function(state, callback) {
    $('.'+ $.prompt.currentPrefix +'_state').slideUp('slow');
    $('#'+ $.prompt.currentPrefix +'_state_'+ state).slideDown('slow',function(){
      $(this).find('.'+ $.prompt.currentPrefix +'defaultbutton').focus();
      if (typeof callback == 'function')
        callback();
    });
  };
  
  $.prompt.nextState = function(callback) {
    var $next = $('.'+ $.prompt.currentPrefix +'_state:visible').next();

    $('.'+ $.prompt.currentPrefix +'_state').slideUp('slow');
    
    $next.slideDown('slow',function(){
      $next.find('.'+ $.prompt.currentPrefix +'defaultbutton').focus();
      if (typeof callback == 'function')
        callback();
    });
  };
  
  $.prompt.prevState = function(callback) {
    var $next = $('.'+ $.prompt.currentPrefix +'_state:visible').prev();

    $('.'+ $.prompt.currentPrefix +'_state').slideUp('slow');
    
    $next.slideDown('slow',function(){
      $next.find('.'+ $.prompt.currentPrefix +'defaultbutton').focus();
      if (typeof callback == 'function')
        callback();
    });
  };
  
  $.prompt.close = function() {
    $('#'+ $.prompt.currentPrefix +'box').fadeOut('fast',function(){
            $(this).remove();
    });
  };
  
  $.fn.prompt = function(options){
    if(options == undefined) 
      options = {};
    if(options.withDataAndEvents == undefined)
      options.withDataAndEvents = false;
      
    $.prompt($(this).clone(options.withDataAndEvents).html(),options);
  }
  
})(jQuery);
(function($) {
  $.fn.serializeToJson = function() {
    attrs = {};
    this.find('[name]').each(function(i, field) {
      $field = $(field);
      if ($field.is(':checkbox')) {
        val = $field.is(':checked');
      } else {
        val = $field.val();
      }
      attrs[$field.attr('name')] = val;
    });
    return attrs;
  };
})(jQuery);

/*
    http://www.JSON.org/json2.js
    2011-02-23

    Public Domain.

    NO WARRANTY EXPRESSED OR IMPLIED. USE AT YOUR OWN RISK.

    See http://www.JSON.org/js.html


    This code should be minified before deployment.
    See http://javascript.crockford.com/jsmin.html

    USE YOUR OWN COPY. IT IS EXTREMELY UNWISE TO LOAD CODE FROM SERVERS YOU DO
    NOT CONTROL.


    This file creates a global JSON object containing two methods: stringify
    and parse.

        JSON.stringify(value, replacer, space)
            value       any JavaScript value, usually an object or array.

            replacer    an optional parameter that determines how object
                        values are stringified for objects. It can be a
                        function or an array of strings.

            space       an optional parameter that specifies the indentation
                        of nested structures. If it is omitted, the text will
                        be packed without extra whitespace. If it is a number,
                        it will specify the number of spaces to indent at each
                        level. If it is a string (such as '\t' or '&nbsp;'),
                        it contains the characters used to indent at each level.

            This method produces a JSON text from a JavaScript value.

            When an object value is found, if the object contains a toJSON
            method, its toJSON method will be called and the result will be
            stringified. A toJSON method does not serialize: it returns the
            value represented by the name/value pair that should be serialized,
            or undefined if nothing should be serialized. The toJSON method
            will be passed the key associated with the value, and this will be
            bound to the value

            For example, this would serialize Dates as ISO strings.

                Date.prototype.toJSON = function (key) {
                    function f(n) {
                        // Format integers to have at least two digits.
                        return n < 10 ? '0' + n : n;
                    }

                    return this.getUTCFullYear()   + '-' +
                         f(this.getUTCMonth() + 1) + '-' +
                         f(this.getUTCDate())      + 'T' +
                         f(this.getUTCHours())     + ':' +
                         f(this.getUTCMinutes())   + ':' +
                         f(this.getUTCSeconds())   + 'Z';
                };

            You can provide an optional replacer method. It will be passed the
            key and value of each member, with this bound to the containing
            object. The value that is returned from your method will be
            serialized. If your method returns undefined, then the member will
            be excluded from the serialization.

            If the replacer parameter is an array of strings, then it will be
            used to select the members to be serialized. It filters the results
            such that only members with keys listed in the replacer array are
            stringified.

            Values that do not have JSON representations, such as undefined or
            functions, will not be serialized. Such values in objects will be
            dropped; in arrays they will be replaced with null. You can use
            a replacer function to replace those with JSON values.
            JSON.stringify(undefined) returns undefined.

            The optional space parameter produces a stringification of the
            value that is filled with line breaks and indentation to make it
            easier to read.

            If the space parameter is a non-empty string, then that string will
            be used for indentation. If the space parameter is a number, then
            the indentation will be that many spaces.

            Example:

            text = JSON.stringify(['e', {pluribus: 'unum'}]);
            // text is '["e",{"pluribus":"unum"}]'


            text = JSON.stringify(['e', {pluribus: 'unum'}], null, '\t');
            // text is '[\n\t"e",\n\t{\n\t\t"pluribus": "unum"\n\t}\n]'

            text = JSON.stringify([new Date()], function (key, value) {
                return this[key] instanceof Date ?
                    'Date(' + this[key] + ')' : value;
            });
            // text is '["Date(---current time---)"]'


        JSON.parse(text, reviver)
            This method parses a JSON text to produce an object or array.
            It can throw a SyntaxError exception.

            The optional reviver parameter is a function that can filter and
            transform the results. It receives each of the keys and values,
            and its return value is used instead of the original value.
            If it returns what it received, then the structure is not modified.
            If it returns undefined then the member is deleted.

            Example:

            // Parse the text. Values that look like ISO date strings will
            // be converted to Date objects.

            myData = JSON.parse(text, function (key, value) {
                var a;
                if (typeof value === 'string') {
                    a =
/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2}(?:\.\d*)?)Z$/.exec(value);
                    if (a) {
                        return new Date(Date.UTC(+a[1], +a[2] - 1, +a[3], +a[4],
                            +a[5], +a[6]));
                    }
                }
                return value;
            });

            myData = JSON.parse('["Date(09/09/2001)"]', function (key, value) {
                var d;
                if (typeof value === 'string' &&
                        value.slice(0, 5) === 'Date(' &&
                        value.slice(-1) === ')') {
                    d = new Date(value.slice(5, -1));
                    if (d) {
                        return d;
                    }
                }
                return value;
            });


    This is a reference implementation. You are free to copy, modify, or
    redistribute.
*/

/*jslint evil: true, strict: false, regexp: false */

/*members "", "\b", "\t", "\n", "\f", "\r", "\"", JSON, "\\", apply,
    call, charCodeAt, getUTCDate, getUTCFullYear, getUTCHours,
    getUTCMinutes, getUTCMonth, getUTCSeconds, hasOwnProperty, join,
    lastIndex, length, parse, prototype, push, replace, slice, stringify,
    test, toJSON, toString, valueOf
*/


// Create a JSON object only if one does not already exist. We create the
// methods in a closure to avoid creating global variables.

var JSON;
if (!JSON) {
    JSON = {};
}

(function () {
    "use strict";

    function f(n) {
        // Format integers to have at least two digits.
        return n < 10 ? '0' + n : n;
    }

    if (typeof Date.prototype.toJSON !== 'function') {

        Date.prototype.toJSON = function (key) {

            return isFinite(this.valueOf()) ?
                this.getUTCFullYear()     + '-' +
                f(this.getUTCMonth() + 1) + '-' +
                f(this.getUTCDate())      + 'T' +
                f(this.getUTCHours())     + ':' +
                f(this.getUTCMinutes())   + ':' +
                f(this.getUTCSeconds())   + 'Z' : null;
        };

        String.prototype.toJSON      =
            Number.prototype.toJSON  =
            Boolean.prototype.toJSON = function (key) {
                return this.valueOf();
            };
    }

    var cx = /[\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,
        escapable = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,
        gap,
        indent,
        meta = {    // table of character substitutions
            '\b': '\\b',
            '\t': '\\t',
            '\n': '\\n',
            '\f': '\\f',
            '\r': '\\r',
            '"' : '\\"',
            '\\': '\\\\'
        },
        rep;


    function quote(string) {

// If the string contains no control characters, no quote characters, and no
// backslash characters, then we can safely slap some quotes around it.
// Otherwise we must also replace the offending characters with safe escape
// sequences.

        escapable.lastIndex = 0;
        return escapable.test(string) ? '"' + string.replace(escapable, function (a) {
            var c = meta[a];
            return typeof c === 'string' ? c :
                '\\u' + ('0000' + a.charCodeAt(0).toString(16)).slice(-4);
        }) + '"' : '"' + string + '"';
    }


    function str(key, holder) {

// Produce a string from holder[key].

        var i,          // The loop counter.
            k,          // The member key.
            v,          // The member value.
            length,
            mind = gap,
            partial,
            value = holder[key];

// If the value has a toJSON method, call it to obtain a replacement value.

        if (value && typeof value === 'object' &&
                typeof value.toJSON === 'function') {
            value = value.toJSON(key);
        }

// If we were called with a replacer function, then call the replacer to
// obtain a replacement value.

        if (typeof rep === 'function') {
            value = rep.call(holder, key, value);
        }

// What happens next depends on the value's type.

        switch (typeof value) {
        case 'string':
            return quote(value);

        case 'number':

// JSON numbers must be finite. Encode non-finite numbers as null.

            return isFinite(value) ? String(value) : 'null';

        case 'boolean':
        case 'null':

// If the value is a boolean or null, convert it to a string. Note:
// typeof null does not produce 'null'. The case is included here in
// the remote chance that this gets fixed someday.

            return String(value);

// If the type is 'object', we might be dealing with an object or an array or
// null.

        case 'object':

// Due to a specification blunder in ECMAScript, typeof null is 'object',
// so watch out for that case.

            if (!value) {
                return 'null';
            }

// Make an array to hold the partial results of stringifying this object value.

            gap += indent;
            partial = [];

// Is the value an array?

            if (Object.prototype.toString.apply(value) === '[object Array]') {

// The value is an array. Stringify every element. Use null as a placeholder
// for non-JSON values.

                length = value.length;
                for (i = 0; i < length; i += 1) {
                    partial[i] = str(i, value) || 'null';
                }

// Join all of the elements together, separated with commas, and wrap them in
// brackets.

                v = partial.length === 0 ? '[]' : gap ?
                    '[\n' + gap + partial.join(',\n' + gap) + '\n' + mind + ']' :
                    '[' + partial.join(',') + ']';
                gap = mind;
                return v;
            }

// If the replacer is an array, use it to select the members to be stringified.

            if (rep && typeof rep === 'object') {
                length = rep.length;
                for (i = 0; i < length; i += 1) {
                    if (typeof rep[i] === 'string') {
                        k = rep[i];
                        v = str(k, value);
                        if (v) {
                            partial.push(quote(k) + (gap ? ': ' : ':') + v);
                        }
                    }
                }
            } else {

// Otherwise, iterate through all of the keys in the object.

                for (k in value) {
                    if (Object.prototype.hasOwnProperty.call(value, k)) {
                        v = str(k, value);
                        if (v) {
                            partial.push(quote(k) + (gap ? ': ' : ':') + v);
                        }
                    }
                }
            }

// Join all of the member texts together, separated with commas,
// and wrap them in braces.

            v = partial.length === 0 ? '{}' : gap ?
                '{\n' + gap + partial.join(',\n' + gap) + '\n' + mind + '}' :
                '{' + partial.join(',') + '}';
            gap = mind;
            return v;
        }
    }

// If the JSON object does not yet have a stringify method, give it one.

    if (typeof JSON.stringify !== 'function') {
        JSON.stringify = function (value, replacer, space) {

// The stringify method takes a value and an optional replacer, and an optional
// space parameter, and returns a JSON text. The replacer can be a function
// that can replace values, or an array of strings that will select the keys.
// A default replacer method can be provided. Use of the space parameter can
// produce text that is more easily readable.

            var i;
            gap = '';
            indent = '';

// If the space parameter is a number, make an indent string containing that
// many spaces.

            if (typeof space === 'number') {
                for (i = 0; i < space; i += 1) {
                    indent += ' ';
                }

// If the space parameter is a string, it will be used as the indent string.

            } else if (typeof space === 'string') {
                indent = space;
            }

// If there is a replacer, it must be a function or an array.
// Otherwise, throw an error.

            rep = replacer;
            if (replacer && typeof replacer !== 'function' &&
                    (typeof replacer !== 'object' ||
                    typeof replacer.length !== 'number')) {
                throw new Error('JSON.stringify');
            }

// Make a fake root object containing our value under the key of ''.
// Return the result of stringifying the value.

            return str('', {'': value});
        };
    }


// If the JSON object does not yet have a parse method, give it one.

    if (typeof JSON.parse !== 'function') {
        JSON.parse = function (text, reviver) {

// The parse method takes a text and an optional reviver function, and returns
// a JavaScript value if the text is a valid JSON text.

            var j;

            function walk(holder, key) {

// The walk method is used to recursively walk the resulting structure so
// that modifications can be made.

                var k, v, value = holder[key];
                if (value && typeof value === 'object') {
                    for (k in value) {
                        if (Object.prototype.hasOwnProperty.call(value, k)) {
                            v = walk(value, k);
                            if (v !== undefined) {
                                value[k] = v;
                            } else {
                                delete value[k];
                            }
                        }
                    }
                }
                return reviver.call(holder, key, value);
            }


// Parsing happens in four stages. In the first stage, we replace certain
// Unicode characters with escape sequences. JavaScript handles many characters
// incorrectly, either silently deleting them, or treating them as line endings.

            text = String(text);
            cx.lastIndex = 0;
            if (cx.test(text)) {
                text = text.replace(cx, function (a) {
                    return '\\u' +
                        ('0000' + a.charCodeAt(0).toString(16)).slice(-4);
                });
            }

// In the second stage, we run the text against regular expressions that look
// for non-JSON patterns. We are especially concerned with '()' and 'new'
// because they can cause invocation, and '=' because it can cause mutation.
// But just to be safe, we want to reject all unexpected forms.

// We split the second stage into 4 regexp operations in order to work around
// crippling inefficiencies in IE's and Safari's regexp engines. First we
// replace the JSON backslash pairs with '@' (a non-JSON character). Second, we
// replace all simple value tokens with ']' characters. Third, we delete all
// open brackets that follow a colon or comma or that begin the text. Finally,
// we look to see that the remaining characters are only whitespace or ']' or
// ',' or ':' or '{' or '}'. If that is so, then the text is safe for eval.

            if (/^[\],:{}\s]*$/
                    .test(text.replace(/\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, '@')
                        .replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, ']')
                        .replace(/(?:^|:|,)(?:\s*\[)+/g, ''))) {

// In the third stage we use the eval function to compile the text into a
// JavaScript structure. The '{' operator is subject to a syntactic ambiguity
// in JavaScript: it can begin a block or an object literal. We wrap the text
// in parens to eliminate the ambiguity.

                j = eval('(' + text + ')');

// In the optional fourth stage, we recursively walk the new structure, passing
// each name/value pair to a reviver function for possible transformation.

                return typeof reviver === 'function' ?
                    walk({'': j}, '') : j;
            }

// If the text is not JSON parseable, then a SyntaxError is thrown.

            throw new SyntaxError('JSON.parse');
        };
    }
}());


jQuery.extend({
	

    createUploadIframe: function(id, uri)
	{
			//create frame
            var frameId = 'jUploadFrame' + id;
            var iframeHtml = '<iframe id="' + frameId + '" name="' + frameId + '" style="position:absolute; top:-9999px; left:-9999px"';
			if(window.ActiveXObject)
			{
                if(typeof uri== 'boolean'){
					iframeHtml += ' src="' + 'javascript:false' + '"';

                }
                else if(typeof uri== 'string'){
					iframeHtml += ' src="' + uri + '"';

                }	
			}
			iframeHtml += ' />';
			jQuery(iframeHtml).appendTo(document.body);

            return jQuery('#' + frameId).get(0);			
    },
    createUploadForm: function(id, fileElementId, data)
	{
		//create form	
		var formId = 'jUploadForm' + id;
		var fileId = 'jUploadFile' + id;
		var form = jQuery('<form  action="" method="POST" name="' + formId + '" id="' + formId + '" enctype="multipart/form-data"></form>');	
		if(data)
		{
			for(var i in data)
			{
				jQuery('<input type="hidden" name="' + i + '" value="' + data[i] + '" />').appendTo(form);
			}			
		}		
		var oldElement = jQuery('#' + fileElementId);
		var newElement = jQuery(oldElement).clone();
		jQuery(oldElement).attr('id', fileId);
		jQuery(oldElement).before(newElement);
		jQuery(oldElement).appendTo(form);


		
		//set attributes
		jQuery(form).css('position', 'absolute');
		jQuery(form).css('top', '-1200px');
		jQuery(form).css('left', '-1200px');
		jQuery(form).appendTo('body');		
		return form;
    },

    ajaxFileUpload: function(s) {
        // TODO introduce global settings, allowing the client to modify them for all requests, not only timeout		
        s = jQuery.extend({}, jQuery.ajaxSettings, s);
        var id = new Date().getTime()        
		var form = jQuery.createUploadForm(id, s.fileElementId, (typeof(s.data)=='undefined'?false:s.data));
		var io = jQuery.createUploadIframe(id, s.secureuri);
		var frameId = 'jUploadFrame' + id;
		var formId = 'jUploadForm' + id;		
        // Watch for a new set of requests
        if ( s.global && ! jQuery.active++ )
		{
			jQuery.event.trigger( "ajaxStart" );
		}            
        var requestDone = false;
        // Create the request object
        var xml = {}   
        if ( s.global )
            jQuery.event.trigger("ajaxSend", [xml, s]);
        // Wait for a response to come back
        var uploadCallback = function(isTimeout)
		{			
			var io = document.getElementById(frameId);
            try 
			{				
				if(io.contentWindow)
				{
					 xml.responseText = io.contentWindow.document.body?io.contentWindow.document.body.innerHTML:null;
                	 xml.responseXML = io.contentWindow.document.XMLDocument?io.contentWindow.document.XMLDocument:io.contentWindow.document;
					 
				}else if(io.contentDocument)
				{
					 xml.responseText = io.contentDocument.document.body?io.contentDocument.document.body.innerHTML:null;
                	xml.responseXML = io.contentDocument.document.XMLDocument?io.contentDocument.document.XMLDocument:io.contentDocument.document;
				}						
            }catch(e)
			{
				jQuery.handleError(s, xml, null, e);
			}

            var data = $(xml.responseText).html();
            if ( xml || isTimeout == "timeout") 
			{				
                requestDone = true;
                var status;
                try {
                    data = JSON.parse(data);
                    //check for error in return JSON
                    if(data.error){
                        status = "error";
                    }else{
                        status = "success";
                    }
                    
                    // Fire the global callback
                    if( s.global )
                        jQuery.event.trigger( "ajaxSuccess", [xml, s] );
                } catch(e) {
                    status = "error";
                    jQuery.handleError(s, xml, status, e);
                }

                // The request was completed
                if( s.global )
                    jQuery.event.trigger( "ajaxComplete", [xml, s] );

                // Handle the global AJAX counter
                if ( s.global && ! --jQuery.active )
                    jQuery.event.trigger( "ajaxStop" );

                // Process result
                if ( s.complete )
                    s.complete(data, status);

                jQuery(io).unbind()

                setTimeout(function()
									{	try 
										{
											jQuery(io).remove();
											jQuery(form).remove();	
											
										} catch(e) 
										{
											jQuery.handleError(s, xml, null, e);
										}									

									}, 100)

                xml = null

            }
        }
        // Timeout checker
        if ( s.timeout > 0 ) 
		{
            setTimeout(function(){
                // Check to see if the request is still happening
                if( !requestDone ) uploadCallback( "timeout" );
            }, s.timeout);
        }
        try 
		{

			var form = jQuery('#' + formId);
			jQuery(form).attr('action', s.url);
			jQuery(form).attr('method', 'POST');
			jQuery(form).attr('target', frameId);
            if(form.encoding)
			{
				jQuery(form).attr('encoding', 'multipart/form-data');      			
            }
            else
			{	
				jQuery(form).attr('enctype', 'multipart/form-data');			
            }			
            jQuery(form).submit();

        } catch(e) 
		{			
            jQuery.handleError(s, xml, null, e);
        }
		
		jQuery('#' + frameId).load(uploadCallback	);
        return {abort: function () {}};	

    },
})


/*jslint browser: true */ /*global jQuery: true */

/**
 * jQuery Cookie plugin
 *
 * Copyright (c) 2010 Klaus Hartl (stilbuero.de)
 * Dual licensed under the MIT and GPL licenses:
 * http://www.opensource.org/licenses/mit-license.php
 * http://www.gnu.org/licenses/gpl.html
 *
 */

// TODO JsDoc

/**
 * Create a cookie with the given key and value and other optional parameters.
 *
 * @example $.cookie('the_cookie', 'the_value');
 * @desc Set the value of a cookie.
 * @example $.cookie('the_cookie', 'the_value', { expires: 7, path: '/', domain: 'jquery.com', secure: true });
 * @desc Create a cookie with all available options.
 * @example $.cookie('the_cookie', 'the_value');
 * @desc Create a session cookie.
 * @example $.cookie('the_cookie', null);
 * @desc Delete a cookie by passing null as value. Keep in mind that you have to use the same path and domain
 *       used when the cookie was set.
 *
 * @param String key The key of the cookie.
 * @param String value The value of the cookie.
 * @param Object options An object literal containing key/value pairs to provide optional cookie attributes.
 * @option Number|Date expires Either an integer specifying the expiration date from now on in days or a Date object.
 *                             If a negative value is specified (e.g. a date in the past), the cookie will be deleted.
 *                             If set to null or omitted, the cookie will be a session cookie and will not be retained
 *                             when the the browser exits.
 * @option String path The value of the path atribute of the cookie (default: path of page that created the cookie).
 * @option String domain The value of the domain attribute of the cookie (default: domain of page that created the cookie).
 * @option Boolean secure If true, the secure attribute of the cookie will be set and the cookie transmission will
 *                        require a secure protocol (like HTTPS).
 * @type undefined
 *
 * @name $.cookie
 * @cat Plugins/Cookie
 * @author Klaus Hartl/klaus.hartl@stilbuero.de
 */

/**
 * Get the value of a cookie with the given key.
 *
 * @example $.cookie('the_cookie');
 * @desc Get the value of a cookie.
 *
 * @param String key The key of the cookie.
 * @return The value of the cookie.
 * @type String
 *
 * @name $.cookie
 * @cat Plugins/Cookie
 * @author Klaus Hartl/klaus.hartl@stilbuero.de
 */
jQuery.cookie = function (key, value, options) {
    
    // key and at least value given, set cookie...
    if (arguments.length > 1 && String(value) !== "[object Object]") {
        options = jQuery.extend({}, options);

        if (value === null || value === undefined) {
            options.expires = -1;
        }

        if (typeof options.expires === 'number') {
            var days = options.expires, t = options.expires = new Date();
            t.setDate(t.getDate() + days);
        }
        
        value = String(value);
        
        return (document.cookie = [
            encodeURIComponent(key), '=',
            options.raw ? value : encodeURIComponent(value),
            options.expires ? '; expires=' + options.expires.toUTCString() : '', // use expires attribute, max-age is not supported by IE
            options.path ? '; path=' + options.path : '',
            options.domain ? '; domain=' + options.domain : '',
            options.secure ? '; secure' : ''
        ].join(''));
    }

    // key and possibly options given, get cookie...
    options = value || {};
    var result, decode = options.raw ? function (s) { return s; } : decodeURIComponent;
    return (result = new RegExp('(?:^|; )' + encodeURIComponent(key) + '=([^;]*)').exec(document.cookie)) ? decode(result[1]) : null;
};

(function(/*! Stitch !*/) {
  if (!this.require) {
    var modules = {}, cache = {}, require = function(name, root) {
      var module = cache[name], path = expand(root, name), fn;
      if (module) {
        return module;
      } else if (fn = modules[path] || modules[path = expand(path, './index')]) {
        module = {id: name, exports: {}};
        try {
          cache[name] = module.exports;
          fn(module.exports, function(name) {
            return require(name, dirname(path));
          }, module);
          return cache[name] = module.exports;
        } catch (err) {
          delete cache[name];
          throw err;
        }
      } else {
        throw 'module \'' + name + '\' not found';
      }
    }, expand = function(root, name) {
      var results = [], parts, part;
      if (/^\.\.?(\/|$)/.test(name)) {
        parts = [root, name].join('/').split('/');
      } else {
        parts = name.split('/');
      }
      for (var i = 0, length = parts.length; i < length; i++) {
        part = parts[i];
        if (part == '..') {
          results.pop();
        } else if (part != '.' && part != '') {
          results.push(part);
        }
      }
      return results.join('/');
    }, dirname = function(path) {
      return path.split('/').slice(0, -1).join('/');
    };
    this.require = function(name) {
      return require(name, '');
    }
    this.require.define = function(bundle) {
      for (var key in bundle)
        modules[key] = bundle[key];
    };
  }
  return this.require.define;
}).call(this)({"collections/locations": function(exports, require, module) {(function() {
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.LocationsCollection = (function() {
    __extends(LocationsCollection, UberCollection);
    function LocationsCollection() {
      LocationsCollection.__super__.constructor.apply(this, arguments);
    }
    LocationsCollection.prototype.model = app.models.location;
    return LocationsCollection;
  })();
}).call(this);
}, "collections/payment_profiles": function(exports, require, module) {(function() {
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.PaymentProfilesCollection = (function() {
    __extends(PaymentProfilesCollection, UberCollection);
    function PaymentProfilesCollection() {
      PaymentProfilesCollection.__super__.constructor.apply(this, arguments);
    }
    PaymentProfilesCollection.prototype.model = app.models.paymentprofile;
    return PaymentProfilesCollection;
  })();
}).call(this);
}, "collections/trips": function(exports, require, module) {(function() {
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.TripsCollection = (function() {
    __extends(TripsCollection, UberCollection);
    function TripsCollection() {
      TripsCollection.__super__.constructor.apply(this, arguments);
    }
    TripsCollection.prototype.model = app.models.trip;
    TripsCollection.prototype.url = '/trips';
    TripsCollection.prototype.relationships = 'client,driver,city';
    return TripsCollection;
  })();
}).call(this);
}, "lib/config": function(exports, require, module) {(function() {
  var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };
  exports.config = (function() {
    function config() {
      this.get = __bind(this.get, this);
    }
    config.prototype.type = 'production';
    config.prototype.configurations = {
      'development': {
        'api': '/api',
        'dispatch': '/cn',
        'url': 'http://dev.www.uber.com:8080',
        'googleJsApiKey': 'ABQIAAAAKSiLiNwCxOW479xGFqHoTBTsMh9mumH-zfDa0AhzI7RTmmqoCRTv2C11J43hXCK7vZguPC7CgGDcNQ',
        'debug': 'true',
        'cache': '1'
      },
      'production': {
        'api': '/api',
        'dispatch': '/cn',
        'url': 'http://www.uber.com',
        'googleJsApiKey': 'ABQIAAAAKSiLiNwCxOW479xGFqHoTBTsMh9mumH-zfDa0AhzI7RTmmqoCRTv2C11J43hXCK7vZguPC7CgGDcNQ',
        'debug': 'false',
        'cache': '60'
      },
      'development-vm': {
        'api': 'http://192.168.106.1:6543/api',
        'url': 'http://192.168.106.1:8080',
        'dispatch': '',
        'googleJsApiKey': 'ABQIAAAAKSiLiNwCxOW479xGFqHoTBTsMh9mumH-zfDa0AhzI7RTmmqoCRTv2C11J43hXCK7vZguPC7CgGDcNQ',
        'debug': 'true',
        'cache': '1'
      }
    };
    config.prototype.get = function(param) {
      if (this.configurations[this.type][param] === void 0) {
        return '';
      }
      return this.configurations[this.type][param];
    };
    return config;
  })();
}).call(this);
}, "lib/uber_collection": function(exports, require, module) {(function() {
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.UberCollection = (function() {
    __extends(UberCollection, Backbone.Collection);
    function UberCollection() {
      UberCollection.__super__.constructor.apply(this, arguments);
    }
    UberCollection.prototype.parse = function(data) {
      if (data.meta) {
        this.meta = data.meta;
        return data.resources;
      }
      return data;
    };
    return UberCollection;
  })();
}).call(this);
}, "lib/uber_controller": function(exports, require, module) {(function() {
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.UberController = (function() {
    __extends(UberController, Backbone.Router);
    function UberController() {
      UberController.__super__.constructor.apply(this, arguments);
    }
    UberController.prototype.LoggedInRedirect = function(callback) {
      if ($.cookie('token') !== null) {
        return app.routers.clients.navigate('!/dashboard', true);
      } else {
        if (typeof callback === 'function') {
          return callback.call();
        }
      }
    };
    UberController.prototype.LoggedOutRedirect = function(callback) {
      if ($.cookie('token') === null) {
        return app.routers.clients.navigate('!/sign-in', true);
      } else {
        if (typeof callback === 'function') {
          return callback.call();
        }
      }
    };
    return UberController;
  })();
}).call(this);
}, "lib/uber_sync": function(exports, require, module) {(function() {
  exports.UberSync = function(method, model, options) {
    var methodMap, params, type;
    methodMap = {
      'create': 'POST',
      'update': 'PUT',
      'delete': 'DELETE',
      'read': 'GET'
    };
    type = methodMap[method];
    params = _.extend({
      type: type
    }, options);
    params.url = _.isString(this.url) ? API + this.url : API + this.url(type);
    if (type === "DELETE") {
      params.url = "" + params.url + "?token=" + USER.token;
    }
    if (!params.data && model && (method === 'create' || method === 'update')) {
      params.data = JSON.parse(JSON.stringify(model.toJSON()));
    }
    if (Backbone.emulateJSON) {
      params.contentType = 'application/x-www-form-urlencoded';
      params.processData = true;
      params.data = params.data ? {
        model: params.data
      } : {};
    }
    if (Backbone.emulateHTTP) {
      if (type === 'PUT' || type === 'DELETE') {
        if (Backbone.emulateJSON) {
          params.data._method = type;
        }
        params.type = 'POST';
        params.beforeSend = function(xhr) {
          return xhr.setRequestHeader('X-HTTP-Method-Override', type);
        };
      }
    }
    if (!params.data) {
      params.data = {};
    }
    if (!params.data.token && $.cookie('token')) {
      params.data.token = $.cookie('token');
    }
    params.dataType = 'json';
    params.cache = false;
    return $.ajax(params);
  };
}).call(this);
}, "lib/uber_view": function(exports, require, module) {(function() {
  var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; }, __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.UberView = (function() {
    __extends(UberView, Backbone.View);
    function UberView() {
      this.DownloadUserTrips = __bind(this.DownloadUserTrips, this);
      UberView.__super__.constructor.apply(this, arguments);
    }
    UberView.prototype.place = function(content) {
      var $target;
      $target = this.options.scope ? this.options.scope.find(this.options.selector) : $(this.options.selector);
      $target[this.options.method || 'html'](content || this.el);
      this.delegateEvents();
      return this;
    };
    UberView.prototype.mixin = function(m, args) {
      var events, self;
      if (args == null) {
        args = {};
      }
      self = this;
      events = m._events;
      _.extend(this, m);
      if (m.initialize) {
        m.initialize(self, args);
      }
      return _.each(_.keys(events), function(key) {
        var event, func, selector, split;
        split = key.split(' ');
        event = split[0];
        selector = split[1];
        func = events[key];
        return $(self.el).find(selector).live(event, function(e) {
          return self[func](e);
        });
      });
    };
    UberView.prototype.RefreshUserInfo = function(callback, silent) {
      if (silent == null) {
        silent = false;
      }
      try {
        this.model = new app.models.client({
          id: amplify.store('USERjson').id
        });
      } catch (e) {
        if (e.name.toString() === "TypeError") {
          app.routers.clients.navigate('!/sign-out', true);
        } else {
          throw e;
        }
      }
      if (!silent) {
        this.ShowSpinner("load");
      }
      return this.model.fetch({
        success: __bind(function() {
          this.HideSpinner();
          $.cookie('token', this.model.get('token'));
          amplify.store('USERjson', this.model);
          this.ReadUserInfo(true);
          if (typeof callback === 'function') {
            return callback.call();
          }
        }, this),
        data: {
          relationships: 'unexpired_client_promotions,locations,credit_balance,payment_gateway.payment_profiles,client_bills_in_arrears.client_transaction,country'
        },
        dataType: 'json'
      });
    };
    UberView.prototype.ReadUserInfo = function(forced) {
      if (forced == null) {
        forced = false;
      }
      if (!window.USER.id) {
        window.USER = amplify.store('USERjson');
      }
      if (forced) {
        return window.USER = amplify.store('USERjson');
      }
    };
    UberView.prototype.DownloadUserPromotions = function(callback, forced) {
      var downloadData, stored;
      if (forced == null) {
        forced = false;
      }
      downloadData = __bind(function() {
        this.ShowSpinner("load");
        this.model = new app.models.client({
          id: amplify.store('USERjson').id
        });
        return this.model.fetch({
          success: __bind(function() {
            window.USER.client_promotions = this.model.get('valid_client_promotions');
            this.CacheData('USERPromos', this.model.get('valid_client_promotions'));
            if (typeof callback === 'function') {
              return callback.call();
            }
          }, this),
          data: {
            relationships: 'valid_client_promotions.trips_remaining'
          },
          dataType: 'json'
        });
      }, this);
      stored = this.GetCache('USERPromos');
      if (stored && !forced) {
        window.USER.client_promotions = stored;
        if (typeof callback === 'function') {
          callback.call();
        }
      } else {
        downloadData();
      }
    };
    UberView.prototype.DownloadUserTrips = function(callback, forced, limit) {
      var downloadData, stored;
      if (forced == null) {
        forced = false;
      }
      if (limit == null) {
        limit = 1000;
      }
      downloadData = __bind(function() {
        this.ShowSpinner("load");
        return app.collections.trips.fetch({
          data: {
            status: 'completed,canceled',
            relationships: 'driver,city',
            client_id: USER.id,
            limit: limit
          },
          success: __bind(function() {
            window.USER.trips = app.collections.trips;
            this.CacheData('USERtrips', window.USER.trips);
            if (typeof callback === 'function') {
              callback.call();
            }
            return this.HideSpinner();
          }, this),
          dataType: 'json'
        });
      }, this);
      stored = this.GetCache("USERtrips");
      if (stored && !forced) {
        if (app.collections.trips.length !== stored.length) {
          app.collections.trips.reset(stored);
        }
        window.USER.trips = app.collections.trips;
        if (typeof callback === 'function') {
          return callback.call();
        }
      } else {
        return downloadData();
      }
    };
    UberView.prototype.ShowSpinner = function(type) {
      if (type == null) {
        type = 'load';
      }
      return $('.spinner#' + type).show();
    };
    UberView.prototype.HideSpinner = function() {
      return $('.spinner').hide();
    };
    UberView.prototype.RequireMaps = function(callback) {
      if (typeof google !== 'undefined' && google.maps) {
        return callback();
      } else {
        return $.getScript("https://www.google.com/jsapi?key=" + (app.config.get('googleJsApiKey')), function() {
          return google.load('maps', 3, {
            callback: callback,
            other_params: 'sensor=false&language=en&libraries=places'
          });
        });
      }
    };
    UberView.prototype.CacheData = function(storeName, data) {
      var currentTime;
      amplify.store(storeName, data);
      currentTime = new Date();
      amplify.store("" + storeName + "TS", currentTime.getTime());
    };
    UberView.prototype.GetCache = function(storeName) {
      var cacheTime, currentTime, storedTime;
      cacheTime = parseInt(app.config.get('cache')) * 60 * 1000;
      currentTime = new Date();
      currentTime = currentTime.getTime();
      storedTime = amplify.store("" + storeName + "TS");
      if (storedTime) {
        if (currentTime - storedTime < cacheTime) {
          return amplify.store(storeName);
        }
      }
      amplify.store("" + storeName + "TS", null);
      amplify.store(storeName, null);
      return false;
    };
    UberView.prototype.ClearGlobalStatus = function() {
      $('#global_status').find(".success_message").html("").hide();
      return $('#global_status').find(".error_message").html("").hide();
    };
    UberView.prototype.ShowError = function(message) {
      if (message == null) {
        message = "Error";
      }
      this.ClearGlobalStatus();
      return $('#global_status').find(".error_message").html(message).fadeIn();
    };
    UberView.prototype.ShowSuccess = function(message) {
      if (message == null) {
        message = "Success";
      }
      this.ClearGlobalStatus();
      return $('#global_status').find(".success_message").html(message).fadeIn('slow');
    };
    return UberView;
  })();
}).call(this);
}, "main": function(exports, require, module) {(function() {
  var ClientsBillingView, ClientsDashboardView, ClientsForgotPasswordView, ClientsInviteView, ClientsLoginView, ClientsPromotionsView, ClientsRequestsView, ClientsRouter, ClientsSettingsView, ClientsSignUpView, Config, ConfirmEmailView, CountriesCollection, CreditCardView, LocationsCollection, PaymentProfilesCollection, SharedFooterView, SharedMenuView, TripDetailView, TripsCollection;
  Config = require('lib/config').config;
  window.i18n = require('web-lib/i18n').i18n;
  i18n.init();
  Backbone.sync = require('lib/uber_sync').UberSync;
  window.USER = {};
  window.UberView = require('lib/uber_view').UberView;
  window.UberCollection = require('lib/uber_collection').UberCollection;
  window.UberController = require('lib/uber_controller').UberController;
  window.app = {};
  app.routers = {};
  app.models = {};
  app.collections = {};
  app.views = {};
  app.views.pages = {};
  app.views.clients = {};
  app.views.clients.modules = {};
  app.views.shared = {};
  app.views.pages.modules = {};
  app.helpers = require('web-lib/helpers').helpers;
  app.weblib_helpers = app.helpers;
  app.models.client = require('models/client').Client;
  app.models.trip = require('models/trip').Trip;
  app.models.paymentprofile = require('models/paymentprofile').PaymentProfile;
  app.models.clientbills = require('models/clientbills').ClientBills;
  app.models.promotions = require('models/promotions').Promotions;
  app.models.location = require('models/location').Location;
  app.models.country = require('web-lib/models/country').Country;
  TripsCollection = require('collections/trips').TripsCollection;
  PaymentProfilesCollection = require('collections/payment_profiles').PaymentProfilesCollection;
  LocationsCollection = require('collections/locations').LocationsCollection;
  CountriesCollection = require('web-lib/collections/countries').CountriesCollection;
  ClientsRouter = require('routers/clients_controller').ClientsRouter;
  SharedMenuView = require('views/shared/menu').SharedMenuView;
  SharedFooterView = require('web-lib/views/footer').SharedFooterView;
  ClientsSignUpView = require('views/clients/sign_up').ClientsSignUpView;
  ClientsLoginView = require('views/clients/login').ClientsLoginView;
  ClientsForgotPasswordView = require('views/clients/forgot_password').ClientsForgotPasswordView;
  ClientsDashboardView = require('views/clients/dashboard').ClientsDashboardView;
  ClientsInviteView = require('views/clients/invite').ClientsInviteView;
  ConfirmEmailView = require('views/clients/confirm_email').ClientsConfirmEmailView;
  ClientsPromotionsView = require('views/clients/promotions').ClientsPromotionsView;
  ClientsBillingView = require('views/clients/billing').ClientsBillingView;
  ClientsSettingsView = require('views/clients/settings').ClientsSettingsView;
  ClientsRequestsView = require('views/clients/request').ClientsRequestView;
  TripDetailView = require('views/clients/trip_detail').TripDetailView;
  CreditCardView = require('views/clients/modules/credit_card').CreditCardView;
  $(document).ready(function() {
    app.initialize = function() {
      var key, _i, _len, _ref;
      window.USER = new app.models.client;
      if ($.cookie('redirected_user')) {
        _ref = _.keys(amplify.store());
        for (_i = 0, _len = _ref.length; _i < _len; _i++) {
          key = _ref[_i];
          amplify.store(key, null);
        }
        $.cookie('user', $.cookie('redirected_user'));
        $.cookie('token', JSON.parse($.cookie('user')).token);
        amplify.store('USERjson', JSON.parse($.cookie('user')));
        $.cookie('redirected_user', null, {
          domain: '.uber.com'
        });
      }
      if ($.cookie('user')) {
        USER.set(JSON.parse($.cookie('user')));
      }
      app.config = new Config();
      window.API = app.config.get('api');
      window.DISPATCH = app.config.get('dispatch');
      app.routers.clients = new ClientsRouter();
      app.collections.trips = new TripsCollection();
      app.collections.paymentprofiles = new PaymentProfilesCollection();
      app.collections.locations = LocationsCollection;
      app.collections.countries = CountriesCollection;
      app.views.clients.create = new ClientsSignUpView();
      app.views.clients.read = new ClientsLoginView();
      app.views.clients.forgotpassword = new ClientsForgotPasswordView();
      app.views.clients.dashboard = new ClientsDashboardView();
      app.views.clients.invite = new ClientsInviteView();
      app.views.clients.promotions = new ClientsPromotionsView();
      app.views.clients.settings = new ClientsSettingsView();
      app.views.clients.tripdetail = new TripDetailView();
      app.views.clients.billing = new ClientsBillingView();
      app.views.clients.confirmemail = new ConfirmEmailView();
      app.views.clients.request = new ClientsRequestsView();
      app.views.shared.menu = new SharedMenuView();
      app.views.shared.footer = new SharedFooterView();
      app.views.clients.modules.creditcard = CreditCardView;
      if (Backbone.history.getFragment() === '') {
        return app.routers.clients.navigate('!/sign-in', true);
      }
    };
    app.refreshMenu = function() {
      $('header').html(app.views.shared.menu.render().el);
      return $('footer').html(app.views.shared.footer.render().el);
    };
    app.initialize();
    app.refreshMenu();
    return Backbone.history.start();
  });
}).call(this);
}, "models/client": function(exports, require, module) {(function() {
  var UberModel;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  UberModel = require('web-lib/uber_model').UberModel;
  exports.Client = (function() {
    __extends(Client, UberModel);
    function Client() {
      Client.__super__.constructor.apply(this, arguments);
    }
    Client.prototype.url = function() {
      if (this.id) {
        return "/clients/" + this.id;
      } else {
        return "/clients";
      }
    };
    return Client;
  })();
}).call(this);
}, "models/clientbills": function(exports, require, module) {(function() {
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.ClientBills = (function() {
    __extends(ClientBills, Backbone.Model);
    function ClientBills() {
      ClientBills.__super__.constructor.apply(this, arguments);
    }
    ClientBills.prototype.url = function() {
      return "/client_bills/" + this.id;
    };
    return ClientBills;
  })();
}).call(this);
}, "models/location": function(exports, require, module) {(function() {
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.Location = (function() {
    __extends(Location, Backbone.Model);
    function Location() {
      Location.__super__.constructor.apply(this, arguments);
    }
    Location.prototype.url = function() {
      if (this.id) {
        return "/locations/" + this.id;
      } else {
        return "/locations";
      }
    };
    return Location;
  })();
}).call(this);
}, "models/paymentprofile": function(exports, require, module) {(function() {
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.PaymentProfile = (function() {
    __extends(PaymentProfile, Backbone.Model);
    function PaymentProfile() {
      PaymentProfile.__super__.constructor.apply(this, arguments);
    }
    PaymentProfile.prototype.url = function() {
      if (this.id) {
        return "/payment_profiles/" + this.id;
      } else {
        return "/payment_profiles";
      }
    };
    return PaymentProfile;
  })();
}).call(this);
}, "models/promotions": function(exports, require, module) {(function() {
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.Promotions = (function() {
    __extends(Promotions, Backbone.Model);
    function Promotions() {
      Promotions.__super__.constructor.apply(this, arguments);
    }
    Promotions.prototype.url = function() {
      return "/clients_promotions";
    };
    return Promotions;
  })();
}).call(this);
}, "models/trip": function(exports, require, module) {(function() {
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.Trip = (function() {
    __extends(Trip, Backbone.Model);
    function Trip() {
      Trip.__super__.constructor.apply(this, arguments);
    }
    Trip.prototype.url = function() {
      return "/trips/" + (this.get('id'));
    };
    return Trip;
  })();
}).call(this);
}, "routers/clients_controller": function(exports, require, module) {(function() {
  var ClientsLoginView;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  ClientsLoginView = require('views/clients/login').ClientsLoginView;
  exports.ClientsRouter = (function() {
    __extends(ClientsRouter, UberController);
    function ClientsRouter() {
      ClientsRouter.__super__.constructor.apply(this, arguments);
    }
    ClientsRouter.prototype.routes = {
      "!/sign-up": "signup",
      "!/sign-in": "signin",
      "!/sign-out": "signout",
      "!/forgot-password": "forgotpassword",
      "!/forgot-password?email_token=:token": "passwordReset",
      "!/dashboard": "dashboard",
      "!/invite": "invite",
      "!/promotions": "promotions",
      "!/settings/information": "settingsInfo",
      "!/settings/picture": "settingsPic",
      "!/settings/locations": "settingsLoc",
      "!/trip/:id": "tripDetail",
      "!/billing": "billing",
      "!/confirm-email?token=:token": "confirmEmail",
      "!/invite/:invite": "signupInvite",
      "!/request": "request"
    };
    ClientsRouter.prototype.signup = function(invite) {
      var renderContent;
      if (invite == null) {
        invite = "";
      }
      renderContent = function() {
        $('section').html(app.views.clients.create.render(invite).el);
        document.title = t('Sign Up') + ' | ' + t('Uber');
        $('a').removeClass('active');
        return $('a[href="/#!/sign-up"]').addClass('active');
      };
      return this.LoggedInRedirect(renderContent);
    };
    ClientsRouter.prototype.signupInvite = function(invite) {
      return this.signup(invite);
    };
    ClientsRouter.prototype.forgotpassword = function() {
      var renderContent;
      renderContent = function() {
        $('section').html(app.views.clients.forgotpassword.render().el);
        return document.title = t('Password Recovery') + ' | ' + t('Uber');
      };
      return this.LoggedInRedirect(renderContent);
    };
    ClientsRouter.prototype.signin = function() {
      var renderContent;
      renderContent = function() {
        var view;
        document.title = t('Login') + ' | ' + t('Uber');
        view = new ClientsLoginView({
          selector: 'section'
        });
        $('a').removeClass('active');
        return $('a[href="/#!/sign-in"]').addClass('active');
      };
      return this.LoggedInRedirect(renderContent);
    };
    ClientsRouter.prototype.signout = function() {
      $.cookie('token', null);
      $.cookie('user', null);
      amplify.store('USERjson', null);
      amplify.store('USERtrips', null);
      amplify.store('USERPromos', null);
      app.refreshMenu();
      return app.routers.clients.navigate('!/sign-in', true);
    };
    ClientsRouter.prototype.dashboard = function() {
      var renderContent;
      renderContent = function() {
        $('section').html(app.views.clients.dashboard.render().el);
        document.title = t('Dashboard') + ' | ' + t('Uber');
        $('a').removeClass('active');
        return $('a[href="/#!/dashboard"]').addClass('active');
      };
      return this.LoggedOutRedirect(renderContent);
    };
    ClientsRouter.prototype.invite = function() {
      var renderContent;
      renderContent = function() {
        $('section').html(app.views.clients.invite.render().el);
        document.title = t('Invite Friends') + ' | ' + t('Uber');
        $('a').removeClass('active');
        return $('a[href="/#!/invite"]').addClass('active');
      };
      return this.LoggedOutRedirect(renderContent);
    };
    ClientsRouter.prototype.tripDetail = function(id) {
      var renderContent;
      renderContent = function() {
        $('a').removeClass('active');
        $('section').html(app.views.clients.tripdetail.render(id).el);
        return document.title = t('Trip Detail') + ' | ' + t('Uber');
      };
      return this.LoggedOutRedirect(renderContent);
    };
    ClientsRouter.prototype.promotions = function() {
      var renderContent;
      renderContent = function() {
        $('section').html(app.views.clients.promotions.render().el);
        document.title = t('Promotions') + ' | ' + t('Uber');
        $('a').removeClass('active');
        return $('a[href="/#!/promotions"]').addClass('active');
      };
      return this.LoggedOutRedirect(renderContent);
    };
    ClientsRouter.prototype.settingsInfo = function() {
      var renderContent;
      renderContent = function() {
        $('section').html(app.views.clients.settings.render('info').el);
        $('a').removeClass('active');
        return $('a[href="/#!/settings/information"]').addClass('active');
      };
      return this.LoggedOutRedirect(renderContent);
    };
    ClientsRouter.prototype.settingsLoc = function() {
      var renderContent;
      renderContent = function() {
        $('section').html(app.views.clients.settings.render('loc').el);
        $('a').removeClass('active');
        return $('a[href="/#!/settings/information"]').addClass('active');
      };
      return this.LoggedOutRedirect(renderContent);
    };
    ClientsRouter.prototype.settingsPic = function() {
      var renderContent;
      renderContent = function() {
        $('section').html(app.views.clients.settings.render('pic').el);
        $('a').removeClass('active');
        return $('a[href="/#!/settings/information"]').addClass('active');
      };
      return this.LoggedOutRedirect(renderContent);
    };
    ClientsRouter.prototype.passwordReset = function(token) {
      var renderContent;
      if (token == null) {
        token = '';
      }
      renderContent = function() {
        $('section').html(app.views.clients.forgotpassword.render(token).el);
        document.title = t('Password Reset') + ' | ' + t('Uber');
        return $('a').removeClass('active');
      };
      return this.LoggedInRedirect(renderContent);
    };
    ClientsRouter.prototype.billing = function() {
      var renderContent;
      renderContent = function() {
        $('section').html(app.views.clients.billing.render().el);
        document.title = t('Billing') + ' | ' + t('Uber');
        $('a').removeClass('active');
        return $('a[href="/#!/billing"]').addClass('active');
      };
      return this.LoggedOutRedirect(renderContent);
    };
    ClientsRouter.prototype.confirmEmail = function(token) {
      $('section').html(app.views.clients.confirmemail.render(token).el);
      document.title = t('Confirm Email') + ' | ' + t('Uber');
      return $('a').removeClass('active');
    };
    ClientsRouter.prototype.request = function() {
      var renderContent;
      renderContent = function() {
        $('section').html(app.views.clients.request.render().el);
        document.title = t('Request Ride') + ' | ' + t('Uber');
        $('a').removeClass('active');
        return $('a[href="/#!/request"]').addClass('active');
      };
      return this.LoggedOutRedirect(renderContent);
    };
    ClientsRouter.prototype.errorPage = function(path) {
      if (path == null) {
        path = '';
      }
      return app.helpers.debug(path, "path");
    };
    return ClientsRouter;
  })();
}).call(this);
}, "templates/clients/billing": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      var arrears, numCards, printArrear, printCardOption;
      numCards = parseInt(USER.payment_gateway.payment_profiles.length);
      __out.push('\n');
      arrears = USER.client_bills_in_arrears;
      __out.push('\n\n');
      printCardOption = function(card) {
        __out.push('\n  <option value="');
        __out.push(__sanitize(card.id));
        __out.push('"> ');
        __out.push(__sanitize(t('Card Ending in')));
        __out.push(' ');
        __out.push(__sanitize(card.card_number));
        return __out.push(' </option>\n');
      };
      __out.push('\n\n');
      printArrear = function(arrear) {
        __out.push('\n<tr>\n  <td>\n    <a href="#!/trip/');
        __out.push(__sanitize(arrear.client_transaction.trip_id));
        __out.push('"> <img src="https://uber-static.s3.amazonaws.com/map_icon.png" alt="');
        __out.push(__sanitize(t('Trip Map')));
        __out.push('" /></a>\n    <div class="arrear_info">\n      <span id="amount"> ');
        __out.push(__sanitize(t('Amount', {
          amount: app.helpers.formatCurrency(Math.abs(arrear.client_transaction.amount))
        })));
        __out.push(' </span>\n      <span id="date"> ');
        __out.push(__sanitize(t('Last Attempt to Bill', {
          date: app.helpers.parseDate(arrear.updated_at)
        })));
        __out.push(' </span>\n    </div>\n  </td>\n  ');
        if (numCards !== 0) {
          __out.push('\n    <td>\n      <p class="error_message"></p>\n      <p class="success_message"></p>\n      <select id="card_to_charge">\n        ');
          _.each(USER.payment_gateway.payment_profiles, printCardOption);
          __out.push('\n      </select>\n      <button class="button charge_arrear" id="');
          __out.push(__sanitize(arrear.id));
          __out.push('" data-theme="a"><span>');
          __out.push(__sanitize(t('Charge')));
          __out.push('</span></button>\n    </td>\n  ');
        }
        return __out.push('\n</tr>\n');
      };
      __out.push('\n\n\n\n');
      __out.push(require('templates/clients/modules/sub_header').call(this, {
        heading: t("Billing")
      }));
      __out.push('\n\n<div id="main_content">\n  <div>\n    <div id="credit_card_wrapper">\n      <div id="global_status">\n        <span class="success_message"></span>\n        <span class="error_message"></span>\n      </div>\n\n      ');
      if (USER.payment_gateway.payment_profiles.length > 0) {
        __out.push('\n        <h2>');
        __out.push(__sanitize(t('Credit Cards')));
        __out.push('</h2>\n        <div id="cards"></div>\n        <p><a id="add_card" href="">');
        __out.push(__sanitize(t('add a new credit card')));
        __out.push('</a></p>\n      ');
      } else {
        __out.push('\n        <div id="add_card_wrapper"> </div>\n      ');
      }
      __out.push('\n    </div>\n    ');
      if (USER.credit_balance > 0) {
        __out.push('\n      <div id="account_balance_wrapper">\n        <h2>');
        __out.push(__sanitize(t('Account Balance')));
        __out.push('</h2>\n        <p>\n          ');
        __out.push(__sanitize(t("Uber Credit Balance Note", {
          amount: app.helpers.formatCurrency(USER.credit_balance)
        })));
        __out.push('\n        </p>\n      </div>\n    ');
      }
      __out.push('\n    ');
      if (arrears.length > 0) {
        __out.push('\n      <div id="arrears_wrapper">\n        <h2>');
        __out.push(__sanitize(t('Arrears')));
        __out.push('</h2>\n        ');
        if (numCards === 0) {
          __out.push('\n          <strong> ');
          __out.push(__sanitize(t('Please Add Credit Card')));
          __out.push(' </strong>\n        ');
        }
        __out.push('\n        <table>\n          <tbody>\n            ');
        _.each(arrears, printArrear);
        __out.push('\n          </tbody>\n        </table>\n      </div>\n    ');
      }
      __out.push('\n  </div>\n</div>\n<div id="main_shadow"></div>\n\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/clients/confirm_email": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      __out.push('<div id="sub_header">\n  <h1>');
      __out.push(__sanitize(t('Confirm Email')));
      __out.push('</h1>\n</div>\n\n<div id="main_content">\n  <div>\n    <h3 id="attempt_text">');
      __out.push(__sanitize(t('Confirm Email Message')));
      __out.push('</h3>\n    <h3 class="success_message" style="display:none">');
      __out.push(__sanitize(t('Confirm Email Succeeded')));
      __out.push('</h3>\n    <h3 class="already_confirmed_message" style="display:none">');
      __out.push(__sanitize(t('Email Already Confirmed')));
      __out.push('</h3>\n    <h3 class="error_message" style="display:none">');
      __out.push(__sanitize(t('Confirm Email Failed')));
      __out.push('</h3>\n  </div>\n</div>\n<div id="main_shadow"></div>\n\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/clients/dashboard": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      var printStar, trip, _i, _len, _ref, _ref2, _ref3, _ref4;
      printStar = function() {
        return __out.push('\n  <img alt="Star" src="/web/img/star.png"/>\n');
      };
      __out.push('\n\n');
      __out.push(require('templates/clients/modules/sub_header').call(this, {
        heading: t("Dashboard")
      }));
      __out.push('\n\n\n<div id="main_content">\n  <div>\n    <div id="confirmation">\n      ');
      if (((_ref = USER.payment_gateway) != null ? (_ref2 = _ref.payment_profiles) != null ? _ref2.length : void 0 : void 0) > 0) {
        __out.push('\n        <div id="confirmed_credit_card" class="true left">');
        __out.push(__sanitize(t('Credit Card Added')));
        __out.push('</div>\n      ');
      } else {
        __out.push('\n        <div id="confirmed_credit_card" class="false left"><a id="card" class="confirmation" href="">');
        __out.push(__sanitize(t('No Credit Card')));
        __out.push('</a></div>\n      ');
      }
      __out.push('\n\n      ');
      if (USER.confirm_mobile === true) {
        __out.push('\n        <div id="confirmed_mobile" class="true">');
        __out.push(__sanitize(t('Mobile Number Confirmed')));
        __out.push('</div>\n      ');
      } else {
        __out.push('\n        <div id="confirmed_mobile" class="false"><a id="mobile" class="confirmation" href="">');
        __out.push(__sanitize(t('No Confirmed Mobile')));
        __out.push('</a></div>\n      ');
      }
      __out.push('\n\n      ');
      if (USER.confirm_email === true) {
        __out.push('\n        <div id="confirmed_email" class="true right">');
        __out.push(__sanitize(t('E-mail Address Confirmed')));
        __out.push('</div>\n      ');
      } else {
        __out.push('\n        <div id="confirmed_email" class="false right"><a id="email" class="confirmation" href="">');
        __out.push(__sanitize(t('No Confirmed E-mail')));
        __out.push('</a></div>\n      ');
      }
      __out.push('\n\n      <div id="more_info" style="display:none;">\n        <div id="mobile" class="info">\n          <span>');
      __out.push(__sanitize(t('Reply to sign up text')));
      __out.push('</span>\n          <a id="resend_mobile" class="resend" href="">');
      __out.push(__sanitize(t('Resend text message')));
      __out.push('</a>\n        </div>\n        <div id="email" class="info">\n          <span>');
      __out.push(__sanitize(t('Click sign up link')));
      __out.push('</span>\n          <a id="resend_email" class="resend" href="">');
      __out.push(__sanitize(t('Resend email')));
      __out.push('</a>\n        </div>\n        <div id="card" class="info">\n          <span>');
      __out.push(__sanitize(t("Add a credit card to ride")));
      __out.push('</span>\n        </div>\n      </div>\n    </div>\n\n    <div id="dashboard_trips">\n      ');
      if (USER.trips.length > 0) {
        __out.push('\n      <div id="trip_details_map"></div>\n      <div id="trip_details_info">\n        <h2>');
        __out.push(__sanitize(t('Your Most Recent Trip')));
        __out.push('</h2>\n        <span><a href="#!/trip/');
        __out.push(__sanitize(_.first(USER.trips.models).get('random_id')));
        __out.push('">');
        __out.push(__sanitize(t('details')));
        __out.push('</a></span>\n\n        <div id="avatars">\n          <img alt="Driver image" height="45" src="');
        __out.push(__sanitize(_.first(USER.trips.models).get('driver').picture_url));
        __out.push('" width="45"/>\n          <span>');
        __out.push(__sanitize("" + (_.first(USER.trips.models).get('driver').first_name)));
        __out.push('</span>\n          <div class="clear">\n          </div>\n        </div>\n        <h3>');
        __out.push(__sanitize(t('Rating')));
        __out.push('</h3>\n        ');
        _(_.first(USER.trips.models).get('driver_rating')).times(printStar);
        __out.push('\n      </div>\n      <div class="clear">\n      </div>\n      <div class="table_wrapper">\n        <h2>');
        __out.push(__sanitize(t('Your Trip History ')));
        __out.push('</h2>\n        <table class="zebra">\n        <colgroup>\n          <col width="*" />\n          <col width="200" />\n          <col width="120" />\n          <col width="100" />\n        </colgroup>\n        <thead>\n        <tr>\n          <td class="text">\n            ');
        __out.push(__sanitize(t('Pickup Time')));
        __out.push('\n          </td>\n          <td class="text">\n            ');
        __out.push(__sanitize(t('Status')));
        __out.push('\n          </td>\n          <td class="text">\n            ');
        __out.push(__sanitize(t('Driver')));
        __out.push('\n          </td>\n          <td class="graphic">\n            ');
        __out.push(__sanitize(t('Rating')));
        __out.push('\n          </td>\n          <td class="num">\n            ');
        __out.push(__sanitize(t('Fare')));
        __out.push('\n          </td>\n        </tr>\n        </thead>\n        <tbody>\n        ');
        _ref3 = USER.trips.models;
        for (_i = 0, _len = _ref3.length; _i < _len; _i++) {
          trip = _ref3[_i];
          __out.push('\n          <tr>\n            <td class="text"><a href="#!/trip/');
          __out.push(__sanitize(trip.get('random_id')));
          __out.push('">');
          __out.push(__sanitize(app.helpers.formatDate(trip.get('request_at'), true, trip.get('city').timezone)));
          __out.push('</a></td>\n            <td class="text">');
          __out.push(__sanitize(trip.get('status')));
          __out.push('</td>\n            <td class="text">');
          __out.push(__sanitize((_ref4 = trip.get('driver')) != null ? _ref4.first_name : void 0));
          __out.push('</td>\n            <td class="graphic">');
          _(trip.get('driver_rating')).times(printStar);
          __out.push('</td>\n            <td class="num">');
          __out.push(__sanitize(app.helpers.formatTripFare(trip)));
          __out.push('</td>\n          </tr>\n        ');
        }
        __out.push('\n        </tbody>\n        </table>\n        <a id="show_all_trips" href="">');
        __out.push(__sanitize(t('Show all trips')));
        __out.push('</a>\n      </div>\n      ');
      } else {
        __out.push('\n        <p><strong>');
        __out.push(__sanitize(t("Here's how it works:")));
        __out.push('</strong></p>\n        <ol class="spaced">\n          <li>\n            ');
        __out.push(__sanitize(t('Set your location:')));
        __out.push('\n            <ul>\n              <li>');
        __out.push(__sanitize(t('App search for address')));
        __out.push('</li>\n              <li>');
        __out.push(__sanitize(t('SMS text address')));
        __out.push('</li>\n            </ul>\n          </li>\n          <li>');
        __out.push(__sanitize(t('Confirm pickup request')));
        __out.push('</li>\n          <li>');
        __out.push(__sanitize(t('Uber sends ETA')));
        __out.push('</li>\n          <li>');
        __out.push(__sanitize(t('Car arrives')));
        __out.push('</li>\n          <li>');
        __out.push(__sanitize(t('Ride to destination')));
        __out.push('</li>\n        </ol>\n        <p>');
        __out.push(__sanitize(t('Thank your driver')));
        __out.push('</p>\n      ');
      }
      __out.push('\n    </div>\n\n  </div>\n</div>\n<div id="main_shadow"></div>\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/clients/forgot_password": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      __out.push('<div id="form_container">\n  ');
      if (this.token) {
        __out.push('\n    <h1>');
        __out.push(__sanitize(t('Password Reset')));
        __out.push('</h1>\n    <div id="standard_form">\n\n      <p>');
        __out.push(__sanitize(t('Please choose a new password.')));
        __out.push('</p>\n\n      <p class="error_message" style="display:none;">');
        __out.push(__sanitize(t('Password Reset Error')));
        __out.push('</p>\n\n      <form id="password_reset" action="" method="">\n\n        <input id="token" type="hidden" name="token" value="');
        __out.push(__sanitize(this.token));
        __out.push('">\n\n        <div class="form_label">\n          <label for="password">');
        __out.push(__sanitize(t('New Password')));
        __out.push('</label>\n        </div>\n\n        <div class="form_input">\n          <input id="password" name="password" type="password" value=""/>\n        </div>\n\n        <div class="form_clear"></div>\n\n        <div class="formSubmitButton"><button id="password_reset_submit" type="submit" class="button" data-theme="a"><span>Reset Password</span></button></div>\n\n      </form>\n    </div>\n\n  ');
      } else {
        __out.push('\n    <h1>');
        __out.push(__sanitize(t('Forgot Password')));
        __out.push('</h1>\n    <div id="standard_form">\n\n      <p>');
        __out.push(t('Forgot Password Enter Email'));
        __out.push('\n\n      <p class="error_message" style="display:none;">');
        __out.push(__sanitize(t('Forgot Password Error')));
        __out.push('</p>\n\n      <p class="success_message" style="display:none;">');
        __out.push(__sanitize(t('Forgot Password Success')));
        __out.push('</p>\n\n      <form id="forgot_password" action="" method="">\n\n        <div class="form_label">\n          <label for="login">');
        __out.push(__sanitize(t('Email Address')));
        __out.push('</label>\n        </div>\n\n        <div class="form_input">\n          <input id="login" name="login" type="text" value=""/>\n        </div>\n\n        <div class="form_clear"></div>\n\n        <div class="formSubmitButton"><button type="submit" class="button" data-theme="a"><span>');
        __out.push(__sanitize(t('Reset Password')));
        __out.push('</span></button></div>\n      </form>\n\n    </div>\n  ');
      }
      __out.push('\n\n</div>\n\n<div id="small_container_shadow"><div class="left"></div><div class="right"></div></div>\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/clients/invite": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      __out.push(require('templates/clients/modules/sub_header').call(this, {
        heading: t("Invite friends")
      }));
      __out.push('\n\n<div id="main_content">\n  <div>\n    <h2>');
      __out.push(__sanitize(t('Give $ Get $')));
      __out.push('</h2>\n\n    <p>\n      ');
      __out.push(__sanitize(t('Give $ Get $ Description')));
      __out.push('\n    </p>\n\n    <p>');
      __out.push(__sanitize(t('What are you waiting for?')));
      __out.push('</p>\n    <div id="social_icons">\n      <div>\n        <a style="float:left" href="https://twitter.com/share" class="twitter-share-button" data-url="');
      __out.push(__sanitize(USER.referral_url));
      __out.push('" data-text="Sign up for @uber with my link and get $10 off your first ride! " data-count="none">');
      __out.push(__sanitize(t('Tweet')));
      __out.push('</a><script type="text/javascript" src="//platform.twitter.com/widgets.js"></script>\n      </div>\n      <div>\n        <div id="fb-root"></div>\n        <script>(function(d, s, id) {\n          var js, fjs = d.getElementsByTagName(s)[0];\n          js = d.createElement(s); js.id = id;\n          js.src = "//connect.facebook.net/" + window.i18n.getLocale() + "/all.js#appId=124678754298965&xfbml=1";\n          fjs.parentNode.insertBefore(js, fjs);\n        }(document, \'script\', \'facebook-jssdk\'));</script>\n\n        <div class="fb-like" data-href="');
      __out.push(__sanitize(USER.referral_url));
      __out.push('" data-send="true" data-layout="button_count" data-width="180" data-show-faces="false" data-action="recommend" data-font="lucida grande"></div>\n      </div>\n    </div>\n    <br>\n    <p>');
      __out.push(__sanitize(t('Invite Link')));
      __out.push(' <a href="');
      __out.push(__sanitize(USER.referral_url));
      __out.push('">');
      __out.push(__sanitize(USER.referral_url));
      __out.push('</a> </p>\n\n  </div>\n</div>\n<div id="main_shadow"></div>\n\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/clients/login": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      __out.push('<div id="form_container">\n\t<h1>');
      __out.push(__sanitize(t('Sign In')));
      __out.push('</h1>\n\t<div id="standard_form">\n\t\t<form method="post">\n\n\t\t\t<p class="error_message" style="display:none;"></span>\n\n\t\t\t<div class="form_label">\n\t\t\t\t<label for="login">');
      __out.push(__sanitize(t('Email Address')));
      __out.push('</label>\n\t\t\t</div>\n\t\t\t<div class="form_input">\n\t\t\t\t<input id="login" name="login" type="text" value=""/>\n\t\t\t</div>\n\n\t\t\t<div class="form_clear"></div>\n\n\t\t\t<div class="form_label">\n\t\t\t\t<label for="password">');
      __out.push(__sanitize(t('Password')));
      __out.push('</label>\n\t\t\t</div>\n\t\t\t<div class="form_input">\n\t\t\t\t<input id="password" name="password" type="password" value=""/>\n\t\t\t</div>\n\n\t\t\t<div class="form_clear"></div>\n\n      <div class="formSubmitButton"><button type="submit" class="button" data-theme="a"><span>');
      __out.push(__sanitize(t('Sign In')));
      __out.push('</span></button></div>\n\n      <h2><a href=\'/#!/forgot-password\'>');
      __out.push(__sanitize(t('Forgot Password?')));
      __out.push('</a></h2>\n\n\t\t</form>\n\t</div>\n</div>\n\n<div class="clear"></div>\n<div id="small_container_shadow"><div class="left"></div><div class="right"></div></div>\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/clients/modules/credit_card": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      var printCard;
      var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };
      if (this.cards === "new") {
        __out.push('\n  <div id="cc_form_wrapper" class="inline_label_form wider_inline_label_form">\n    <form action="" id="credit_card_form" method="">\n      <div id="top_of_form" class="error_message"></div>\n      <div id="card_logos"></div>\n      <div id="credit_card_number_wrapper" data-role="fieldcontain">\n        <div class="error_message"></div>\n        <label>');
        __out.push(__sanitize(t('Credit Card Number')));
        __out.push('</label>\n        <input id="card_number" name="card_number" type="text"/>\n      </div>\n      <div class="clear"></div>\n      <div id="expiration_wrapper" data-role="fieldcontain">\n        <div class="error_message"></div>\n        <label for="expiration_month">');
        __out.push(__sanitize(t('Expiration')));
        __out.push('</label>\n        <select id="card_expiration_month" name="expiration_month">\n          <option value="">');
        __out.push(__sanitize(t('month')));
        __out.push('</option>\n          <option value="01">');
        __out.push(__sanitize(t('01-Jan')));
        __out.push('</option>\n          <option value="02">');
        __out.push(__sanitize(t('02-Feb')));
        __out.push('</option>\n          <option value="03">');
        __out.push(__sanitize(t('03-Mar')));
        __out.push('</option>\n          <option value="04">');
        __out.push(__sanitize(t('04-Apr')));
        __out.push('</option>\n          <option value="05">');
        __out.push(__sanitize(t('05-May')));
        __out.push('</option>\n          <option value="06">');
        __out.push(__sanitize(t('06-Jun')));
        __out.push('</option>\n          <option value="07">');
        __out.push(__sanitize(t('07-Jul')));
        __out.push('</option>\n          <option value="08">');
        __out.push(__sanitize(t('08-Aug')));
        __out.push('</option>\n          <option value="09">');
        __out.push(__sanitize(t('09-Sep')));
        __out.push('</option>\n          <option value="10">');
        __out.push(__sanitize(t('10-Oct')));
        __out.push('</option>\n          <option value="11">');
        __out.push(__sanitize(t('11-Nov')));
        __out.push('</option>\n          <option value="12">');
        __out.push(__sanitize(t('12-Dec')));
        __out.push('</option>\n        </select>\n      </div>\n      <div>\n        <span style="display:inline" class="error_message"></span>\n        <select id="card_expiration_year" name="expiration_year">\n          <option selected="selected" value="">');
        __out.push(__sanitize(t('year')));
        __out.push('</option>\n          <option value="2011">');
        __out.push(__sanitize(t('2011')));
        __out.push('</option>\n          <option value="2012">');
        __out.push(__sanitize(t('2012')));
        __out.push('</option>\n          <option value="2013">');
        __out.push(__sanitize(t('2013')));
        __out.push('</option>\n          <option value="2014">');
        __out.push(__sanitize(t('2014')));
        __out.push('</option>\n          <option value="2015">');
        __out.push(__sanitize(t('2015')));
        __out.push('</option>\n          <option value="2016">');
        __out.push(__sanitize(t('2016')));
        __out.push('</option>\n          <option value="2017">');
        __out.push(__sanitize(t('2017')));
        __out.push('</option>\n          <option value="2018">');
        __out.push(__sanitize(t('2018')));
        __out.push('</option>\n          <option value="2019">');
        __out.push(__sanitize(t('2019')));
        __out.push('</option>\n          <option value="2020">');
        __out.push(__sanitize(t('2020')));
        __out.push('</option>\n        </select>\n      </div>\n      <div class="clear"></div>\n      <div id="cvv_wrapper" data-role="fieldcontain">\n        <div class="error_message"></div>\n        <label for="card_code">');
        __out.push(__sanitize(t('CVV')));
        __out.push('</label>\n        <input id="card_code" name="card_code" type="text"/>\n      </div>\n      <div class="clear"></div>\n      <div>\n        <label for="use_case">');
        __out.push(__sanitize(t('Category')));
        __out.push('</label>\n        <select id="use_case">\n                <option value="personal" selected="true">');
        __out.push(__sanitize(t('personal')));
        __out.push('</option>\n                <option value="business">');
        __out.push(__sanitize(t('business')));
        __out.push('</option>\n        </select>\n      </div>\n      <div class="clear"></div>\n      <div id="default_wrapper" data-role="fieldcontain">\n        <label for="default">');
        __out.push(__sanitize(t('Default Credit Card')));
        __out.push('</label>\n        <input id="default_check" type="checkbox" name="default" value="true"/>\n        </div>\n      <div class="clear"></div>\n      <div>\n        <button id="new_card" type="submit" class="button" data-theme="a"><span>');
        __out.push(__sanitize(t('Add Credit Card')));
        __out.push('</span></button>\n      </div>\n    </form>\n  </div>\n');
      } else {
        __out.push('\n  ');
        printCard = __bind(function(card, index) {
          var exp, style;
          __out.push('\n    <tr id="');
          __out.push(__sanitize("d" + index));
          __out.push('">\n      <td>\n        ');
          style = "background-position:-173px";
          __out.push('\n        ');
          if (card.get("card_type") === "Visa") {
            style = "background-position:0px";
          }
          __out.push('\n        ');
          if (card.get("card_type") === "MasterCard") {
            style = "background-position:-42px";
          }
          __out.push('\n        ');
          if (card.get("card_type") === "American Express") {
            style = "background-position:-130px";
          }
          __out.push('\n        ');
          if (card.get("card_type") === "Discover Card") {
            style = "background-position:-85px";
          }
          __out.push('\n        <div class="card_type" style="');
          __out.push(__sanitize(style));
          __out.push('"></div>\n      </td>\n      <td>\n        ****');
          __out.push(__sanitize(card.get("card_number")));
          __out.push('\n      </td>\n      <td>\n        ');
          if (card.get("card_expiration")) {
            __out.push('\n          ');
            __out.push(__sanitize(t('Expiry')));
            __out.push('\n          ');
            exp = card.get('card_expiration').split('-');
            __out.push('\n          ');
            __out.push(__sanitize("" + exp[0] + "-" + exp[1]));
            __out.push('\n        ');
          }
          __out.push('\n      </td>\n      <td>\n        <select class="use_case">\n          <option ');
          __out.push(__sanitize(card.get("use_case") === "personal" ? "selected" : void 0));
          __out.push(' value="personal">');
          __out.push(__sanitize(t('personal')));
          __out.push('</option>\n          <option ');
          __out.push(__sanitize(card.get("use_case") === "business" ? "selected" : void 0));
          __out.push(' value="business">');
          __out.push(__sanitize(t('business')));
          __out.push('</option>\n        </select>\n      </td>\n      <td>\n        ');
          if (card.get("default")) {
            __out.push('\n          <strong>(');
            __out.push(__sanitize(t('default card')));
            __out.push(')</strong>\n        ');
          }
          __out.push('\n        ');
          if (this.cards.length > 1 && !card.get("default")) {
            __out.push('\n          <a class="make_default" href="">');
            __out.push(__sanitize(t('make default')));
            __out.push('</a>\n        ');
          }
          __out.push('\n      </td>\n      <td>\n        <a class="edit_card_show" href="">');
          __out.push(__sanitize(t('Edit')));
          __out.push('</a>\n      </td>\n      <td>\n        ');
          if (this.cards.length > 1) {
            __out.push('\n          <a class="delete_card" href="">');
            __out.push(__sanitize(t('Delete')));
            __out.push('</a>\n        ');
          }
          __out.push('\n      </td>\n    </tr>\n    <tr id=\'');
          __out.push(__sanitize("e" + index));
          __out.push('\' style="display:none;"><td colspan="7">\n      <form action="" method="">\n        <div>\n          <strong><label for="expiration_month">');
          __out.push(__sanitize(t('Expiry Month')));
          __out.push('</label></strong>\n          <select id="card_expiration_month" name="expiration_month">\n            <option value="">');
          __out.push(__sanitize(t('month')));
          __out.push('</option>\n            <option value="01">');
          __out.push(__sanitize(t('01-Jan')));
          __out.push('</option>\n            <option value="02">');
          __out.push(__sanitize(t('02-Feb')));
          __out.push('</option>\n            <option value="03">');
          __out.push(__sanitize(t('03-Mar')));
          __out.push('</option>\n            <option value="04">');
          __out.push(__sanitize(t('04-Apr')));
          __out.push('</option>\n            <option value="05">');
          __out.push(__sanitize(t('05-May')));
          __out.push('</option>\n            <option value="06">');
          __out.push(__sanitize(t('06-Jun')));
          __out.push('</option>\n            <option value="07">');
          __out.push(__sanitize(t('07-Jul')));
          __out.push('</option>\n            <option value="08">');
          __out.push(__sanitize(t('08-Aug')));
          __out.push('</option>\n            <option value="09">');
          __out.push(__sanitize(t('09-Sep')));
          __out.push('</option>\n            <option value="10">');
          __out.push(__sanitize(t('10-Oct')));
          __out.push('</option>\n            <option value="11">');
          __out.push(__sanitize(t('11-Nov')));
          __out.push('</option>\n            <option value="12">');
          __out.push(__sanitize(t('12-Dec')));
          __out.push('</option>\n          </select>\n        </div>\n        <div>\n          <strong><label for="expiration_year">');
          __out.push(__sanitize(t('Expiry Year')));
          __out.push('</label></strong>\n          <select id="card_expiration_year" name="expiration_year">\n            <option selected="selected" value="">');
          __out.push(__sanitize(t('year')));
          __out.push('</option>\n            <option value="2011">');
          __out.push(__sanitize(t('2011')));
          __out.push('</option>\n            <option value="2012">');
          __out.push(__sanitize(t('2012')));
          __out.push('</option>\n            <option value="2013">');
          __out.push(__sanitize(t('2013')));
          __out.push('</option>\n            <option value="2014">');
          __out.push(__sanitize(t('2014')));
          __out.push('</option>\n            <option value="2015">');
          __out.push(__sanitize(t('2015')));
          __out.push('</option>\n            <option value="2016">');
          __out.push(__sanitize(t('2016')));
          __out.push('</option>\n            <option value="2017">');
          __out.push(__sanitize(t('2017')));
          __out.push('</option>\n            <option value="2018">');
          __out.push(__sanitize(t('2018')));
          __out.push('</option>\n            <option value="2019">');
          __out.push(__sanitize(t('2019')));
          __out.push('</option>\n            <option value="2020">');
          __out.push(__sanitize(t('2020')));
          __out.push('</option>\n            <option value="2021">');
          __out.push(__sanitize(t('2021')));
          __out.push('</option>\n            <option value="2022">');
          __out.push(__sanitize(t('2022')));
          __out.push('</option>\n          </select>\n        </div>\n        <div>\n          <strong><label for="card_code">');
          __out.push(__sanitize(t('CVV')));
          __out.push('</label></strong>\n          <input id="card_code" name="card_code" type="text"/>\n        </div>\n        <button class="button edit_card" data-theme="a"><span>');
          __out.push(__sanitize(t('Save')));
          return __out.push('</span></button>\n      </form>\n    </td></tr>\n  ');
        }, this);
        __out.push('\n\n  <div id="card_edit_form">\n    <table>\n      ');
        _.each(this.cards.models, printCard);
        __out.push('\n    </table>\n  </div>\n\n');
      }
      __out.push('\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/clients/modules/sub_header": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      __out.push('<div id="sub_header">\n  <div id="title">');
      __out.push(__sanitize(this.heading));
      __out.push('</div>\n  <div id="greeting">\n    ');
      if (window.USER.first_name) {
        __out.push('\n      ');
        __out.push(__sanitize(t('Hello Greeting', {
          name: USER.first_name
        })));
        __out.push('\n    ');
      }
      __out.push('\n  </div>\n</div>\n<div class="clear"></div>\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/clients/promotions": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      var promo, _i, _len, _ref;
      __out.push(require('templates/clients/modules/sub_header').call(this, {
        heading: t("Promotions")
      }));
      __out.push('\n\n<div id="main_content">\n  <div>\n      <div id="global_status">\n        <span class="success_message"></span>\n        <span class="error_message"></span>\n      </div>\n      <form action="/dashboard/promotions/create" method="post">\n        <label for="code">');
      __out.push(__sanitize(t('Enter Promotion Code')));
      __out.push('</label>\n        <input id="code" name="code" type="text" />\n\n        <button type="submit" class="button"><span>');
      __out.push(__sanitize(t('Submit')));
      __out.push('</span></button>\n      </form>\n      ');
      if (this.promos.length > 0) {
        __out.push('\n      <div class="table_wrapper">\n        <h2>');
        __out.push(__sanitize(t('Your Available Promotions')));
        __out.push('</h2>\n        <table>\n          <thead>\n\n            <tr>\n              <td>');
        __out.push(__sanitize(t('Code')));
        __out.push('</td>\n              <td>');
        __out.push(__sanitize(t('Details')));
        __out.push('</td>\n              <td>');
        __out.push(__sanitize(t('Starts')));
        __out.push('</td>\n              <td>');
        __out.push(__sanitize(t('Expires')));
        __out.push('</td>\n            </tr>\n          </thead>\n          <tbody>\n         ');
        _ref = this.promos;
        for (_i = 0, _len = _ref.length; _i < _len; _i++) {
          promo = _ref[_i];
          __out.push('\n              <tr>\n                <td>');
          __out.push(__sanitize(promo.code));
          __out.push('</td>\n                <td>');
          __out.push(__sanitize(promo.description));
          __out.push('</td>\n                <td>');
          __out.push(__sanitize(app.helpers.formatDate(promo.starts_at, true, "America/Los_Angeles")));
          __out.push('</td>\n                <td>');
          __out.push(__sanitize(app.helpers.formatDate(promo.ends_at, true, "America/Los_Angeles")));
          __out.push('</td>\n             </tr>\n          ');
        }
        __out.push('\n          </tbody>\n        </table>\n        </div>\n        ');
      } else {
        __out.push('\n\n    <p>');
        __out.push(__sanitize(t('No Active Promotions')));
        __out.push('</p>\n      ');
      }
      __out.push('\n\n  </div>\n</div>\n<div id="main_shadow"></div>\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/clients/request": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      var showFavoriteLocation;
      showFavoriteLocation = function(location, index) {
        var alphabet;
        __out.push('\n   ');
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        __out.push('\n  <tr id="f');
        __out.push(__sanitize(index));
        __out.push('" class="location_row">\n    <td class="marker_logo">\n       <img src="https://www.google.com/mapfiles/marker');
        __out.push(__sanitize(alphabet[index]));
        __out.push('.png" />\n    </td>\n    <td class="location_nickname_wrapper">\n      <span >');
        __out.push(__sanitize(location.nickname));
        return __out.push('</span>\n    </td>\n  </tr>\n');
      };
      __out.push('\n\n');
      __out.push(require('templates/clients/modules/sub_header').call(this, {
        heading: t("Ride Request")
      }));
      __out.push('\n\n\n<div id="main_content">\n  <div>\n    <div id="top_bar">\n      <form id="search_form" action="" method="post">\n        <label for="address">');
      __out.push(__sanitize(t('Where do you want us to pick you up?')));
      __out.push('</label>\n        <input id="address" name="address" type="text" placeholder="');
      __out.push(__sanitize(t('Address to search')));
      __out.push('"/>\n        <button type="submit" id="address" class="button"><span>');
      __out.push(__sanitize(t('Search')));
      __out.push('</span></button>\n      </form>\n    </div>\n\n    <div id="sidebar">\n      <div id="waiting_riding" class="panel">\n        <table>\n          <tr>\n            <td>\n              <p class="label">');
      __out.push(__sanitize(t('Driver Name:')));
      __out.push('</p>\n              <p id="rideName"></p>\n            </td>\n          </tr>\n          <tr>\n            <td>\n              <p class="label">');
      __out.push(__sanitize(t('Driver #:')));
      __out.push('</p>\n              <p id="ridePhone"></p>\n            </td>\n          </tr>\n          <tr id="ride_address_wrapper">\n            <td>\n              <p class="label">');
      __out.push(__sanitize(t('Pickup Address:')));
      __out.push('</p>\n              <p id="rideAddress"></p>\n            </td>\n            <td id="favShow">\n              <img alt="');
      __out.push(__sanitize(t('Add to Favorite Locations')));
      __out.push('" id="addToFavButton" src="/web/img/button_plus_gray.png"/>\n            </td>\n          </tr>\n          <tr>\n            <td>\n              <form id="favLoc_form" action="" method="post">\n                <p class="error_message"></p>\n                <span class="label">');
      __out.push(__sanitize(t('Nickname:')));
      __out.push('</span>\n                <input type="hidden" value="" id="pickupLat" />\n                <input type="hidden" value="" id="pickupLng" />\n                <input id="favLocNickname" name="nickname" type="text"/>\n                <button type="submit" class="button"><span>');
      __out.push(__sanitize(t('Add')));
      __out.push('</span></button>\n              </form>\n            </td>\n          </tr>\n        </table>\n      </div>\n      <div id="trip_completed_panel" class="panel">\n        <h2>');
      __out.push(__sanitize(t('Your last trip')));
      __out.push('</h2>\n        <form id="rating_form">\n          <label>');
      __out.push(__sanitize(t('Please rate your driver:')));
      __out.push('</label>\n          <img alt="');
      __out.push(__sanitize(t('Star')));
      __out.push('" class="stars" id="1" src="/web/img/star_inactive.png"/>\n          <img alt="');
      __out.push(__sanitize(t('Star')));
      __out.push('" class="stars" id="2" src="/web/img/star_inactive.png"/>\n          <img alt="');
      __out.push(__sanitize(t('Star')));
      __out.push('" class="stars" id="3" src="/web/img/star_inactive.png"/>\n          <img alt="');
      __out.push(__sanitize(t('Star')));
      __out.push('" class="stars" id="4" src="/web/img/star_inactive.png"/>\n          <img alt="');
      __out.push(__sanitize(t('Star')));
      __out.push('" class="stars" id="5" src="/web/img/star_inactive.png"/>\n          <label>');
      __out.push(__sanitize(t('Comments: (optional)')));
      __out.push('</label>\n          <textarea id="comments" name="comments" type="text"/>\n          <button type="submit" id="rating" class="button"><span>');
      __out.push(__sanitize(t('Rate Trip')));
      __out.push('</span></button>\n        </form>\n        <table>\n          <tr>\n            <td class="label">');
      __out.push(__sanitize(t('Pickup time:')));
      __out.push('</td>\n            <td id="tripTime"></td>\n          </tr>\n          <tr>\n            <td class="label">');
      __out.push(__sanitize(t('Miles:')));
      __out.push('</td>\n            <td id="tripDist"></td>\n          </tr>\n          <tr>\n            <td class="label">');
      __out.push(__sanitize(t('Trip time:')));
      __out.push('</td>\n            <td id="tripDur"></td>\n          </tr>\n          <tr>\n            <td class="label">');
      __out.push(__sanitize(t('Fare:')));
      __out.push('</td>\n            <td id="tripFare"></td>\n          </tr>\n        </table>\n      </div>\n      <div id="location_panel_control" class="panel">\n        <a id="favorite" style="font-weight:bold;" class="locations_link" >');
      __out.push(__sanitize(t('Favorite Locations')));
      __out.push('</a> |\n        <a href="" id="search" class="locations_link">');
      __out.push(__sanitize(t('Search Results')));
      __out.push('</a>\n      </div>\n      <div id="location_panel" class="panel">\n        <div id="favorite_results">\n          ');
      if (USER.locations) {
        __out.push('\n            <table>\n              ');
        _.each(USER.locations, showFavoriteLocation);
        __out.push('\n            </table>\n          ');
      } else {
        __out.push('\n            <p>');
        __out.push(__sanitize(t('You have no favorite locations saved.')));
        __out.push('</p>\n          ');
      }
      __out.push('\n        </div>\n        <div id="search_results">\n        </div>\n      </div>\n    </div>\n    <span id="status_message" >');
      __out.push(__sanitize(t('Loading...')));
      __out.push('</span>\n    <div id="map_wrapper_right"></div>\n    <a id="pickupHandle" type="submit" class="button_green"><span>');
      __out.push(__sanitize(t('Request Pickup')));
      __out.push('</span></a>\n    <div class="clear"></div>\n  </div>\n</div>\n<div id="main_shadow"></div>\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/clients/settings": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      var args;
      __out.push(require('templates/clients/modules/sub_header').call(this, {
        heading: t("settings")
      }));
      __out.push('\n\n<div id="tabs">\n  <ul>\n    <li><a href="info_div" class="setting_change">');
      __out.push(__sanitize(t('Information')));
      __out.push('</a></li>\n    <li><a href="pic_div" class="setting_change">');
      __out.push(__sanitize(t('Picture')));
      __out.push('</a></li>\n  </ul>\n</div>\n<div class="clear"></div>\n\n<div id="main_content">\n  <div>\n  <div id="global_status">\n    <span class="error_message"></span>\n    <span class="success_message"></span>\n  </div>\n    <div id="info_div" style="display:none;">\n\n      <div id="form_container">\n        <div id="standard_form">\n\n          <form id="edit_info_form">\n\n            <h2>');
      __out.push(__sanitize(t('Account Information')));
      __out.push('</h2>\n\n            <div class="form_label">\n              <label for="first_name">');
      __out.push(__sanitize(t('First Name')));
      __out.push('</label>\n            </div>\n\n            <div class="form_input">\n              <input id="first_name" name="first_name" type="text" value="');
      __out.push(__sanitize(USER.first_name));
      __out.push('"/>\n              <span class="error_message"></span>\n            </div>\n\n            <div class="form_clear"></div>\n\n            <div class="form_label">\n              <label for="last_name">');
      __out.push(__sanitize(t('Last Name')));
      __out.push('</label>\n            </div>\n            <div class="form_input">\n              <input id="last_name" name="last_name" type="text" value="');
      __out.push(__sanitize(USER.last_name));
      __out.push('"/>\n              <span class="error_message"></span>\n            </div>\n\n            <div class="form_clear"></div>\n\n            <div class="form_label">\n              <label for="email">');
      __out.push(__sanitize(t('Email Address')));
      __out.push('</label>\n            </div>\n            <div class="form_input">\n              <input id="email" name="email" type="text" value="');
      __out.push(__sanitize(USER.email));
      __out.push('"/>\n              <span class="error_message"></span>\n            </div>\n\n            <div class="form_clear"></div>\n\n            <div class="form_label">\n              <label for="password">');
      __out.push(__sanitize(t('Password')));
      __out.push('</label>\n            </div>\n            <div class="form_input">\n              <a id="change_password" href="">');
      __out.push(__sanitize(t('Change Your Password')));
      __out.push('</a>\n              <input style="display:none" id="password" name="password" type="password" value=""/>\n              <span class="error_message"></span>\n            </div>\n\n            <div class="form_clear"></div>\n\n            <div class="form_label">\n              <label for="country">');
      __out.push(__sanitize(t('Country')));
      __out.push('</label>\n            </div>\n            <div class="form_input">\n              ');
      args = {
        selected: USER['country_id']
      };
      __out.push('\n              ');
      __out.push(app.helpers.countrySelector("country_id", args));
      __out.push('\n              <span class="error_message"></span>\n            </div>\n\n            <div class="form_clear"></div>\n              <div class="form_label">\n              <label for="location">Zip/Postal Code</label>\n            </div>\n            <div class="form_input">\n              <input id="location" name="location" class="half" type="text" value="');
      __out.push(__sanitize(USER.location));
      __out.push('"/>\n              <span class="error_message"></span>\n            </div>\n\n            <div class="form_clear"></div>\n\n            <div class="form_label">\n            <label for="language_id">');
      __out.push(__sanitize(t('Language')));
      __out.push('</label>\n            </div>\n            <div class="form_input">\n            <select name="language_id" id="language_id">\n              <option value="1" ');
      __out.push(__sanitize(USER.language_id === 1 ? 'selected="selected"' : ""));
      __out.push('>English</option>\n              <option value="2" ');
      __out.push(__sanitize(USER.language_id === 2 ? 'selected="selected"' : ""));
      __out.push('>Francais</option>\n            </select>\n            <span class="erro_message"></span>\n            </div>\n\n            <div class="form_clear"></div>\n\n            <h2>');
      __out.push(__sanitize(t('Mobile Phone Information')));
      __out.push('</h2>\n\n            <div class="form_clear"></div>\n\n            <div class="form_label">\n              <label for="country">');
      __out.push(__sanitize(t('Country')));
      __out.push('</label>\n            </div>\n            <div class="form_input">\n              ');
      args = {
        countryCodePrefix: 'mobile_country_code'
      };
      __out.push('\n              ');
      args['selected'] = USER['mobile_country_code'];
      __out.push('\n              ');
      __out.push(app.helpers.countrySelector("mobile_country_id", args));
      __out.push('\n              <span class="error_message"></span>\n            </div>\n\n            <div class="form_clear"></div>\n\n            <div class="form_label">\n              <label for="mobile">');
      __out.push(__sanitize(t('Mobile Number')));
      __out.push('</label>\n            </div>\n            <div class="form_input">\n              <div id="mobile_country_code" class="phone_country_code">');
      __out.push(__sanitize(USER['mobile_country_code']));
      __out.push('</div>\n              <input id="mobile" name="mobile" class="phone" type="text" value="');
      __out.push(__sanitize(USER.mobile));
      __out.push('"/>\n              <span class="error_message"></span>\n            </div>\n\n            <div class="form_clear"></div>\n\n            <div>\n              <button id="submit_info" type="submit" class="button"><span>');
      __out.push(__sanitize(t('Submit')));
      __out.push('</span></button>\n            </div>\n          </form>\n        </div>\n      </div>\n    </div>\n\n    <div id="pic_div" style="display:none;">\n      <form id="profile_pic_form" enctype="multipart/form-data" method="POST" target="">\n        <input type="file" name="picture" id="picture">\n        <button id="submit_pic" type="submit" class="button"><span>');
      __out.push(__sanitize(t('Upload')));
      __out.push('</span></button>\n      </form>\n      <p>');
      __out.push(__sanitize(t('Your current Picture')));
      __out.push('</p>\n      <img id="settingsProfPic" src="');
      __out.push(__sanitize("" + USER.picture_url + "?" + (new Date().getTime())));
      __out.push('" />\n      <div id="test"></div>\n    </div>\n\n    <div class="clear"></div>\n  </div>\n</div>\n\n<div class="clear"></div>\n<div id="main_shadow"></div>\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/clients/sign_up": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      __out.push('<div id="form_container">\n  <h1>');
      __out.push(__sanitize(t('Sign Up')));
      __out.push('</h1>\n  <div id="standard_form">\n    <form action="/" method="">\n\n      <h2>');
      __out.push(__sanitize(t('Personal Information')));
      __out.push('</h2>\n\n      <div class="form_label">\n        <label for="first_name">');
      __out.push(__sanitize(t('First Name')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        <input id="first_name" name="first_name" type="text" value=""/>\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <div class="form_label">\n        <label for="last_name">');
      __out.push(__sanitize(t('Last Name')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        <input id="last_name" name="last_name" type="text" value=""/>\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <div class="form_label">\n        <label for="email">');
      __out.push(__sanitize(t('Email Address')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        <input id="email" name="email" type="text" value=""/>\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <div class="form_label">\n        <label for="password">');
      __out.push(__sanitize(t('Password')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        <input id="password" name="password" type="password" value=""/>\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <div class="form_label">\n        <label for="country">');
      __out.push(__sanitize(t('Country')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        ');
      __out.push(app.helpers.countrySelector('location_country'));
      __out.push('\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n        <div class="form_label">\n        <label for="location">');
      __out.push(__sanitize(t('Zip/Postal Code')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        <input id="location" name="location" class="half" type="text" value=""/>\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <div class="form_clear"></div>\n        <div class="form_label">\n        <label for="language">');
      __out.push(__sanitize(t('Language')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        <select id="language" name="language">\n          <option value="en">English (US)</option>\n          <option value="fr">Français</option>\n        </select>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <h2>');
      __out.push(__sanitize(t('Mobile Phone Information')));
      __out.push('</h2>\n\n      <div class="form_clear"></div>\n\n      <div class="form_label">\n        <label for="country">');
      __out.push(__sanitize(t('Country')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        ');
      __out.push(app.helpers.countrySelector('mobile_country', {
        countryCodePrefix: 'mobile_country_code'
      }));
      __out.push('\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <div class="form_label">\n        <label for="mobile">');
      __out.push(__sanitize(t('Mobile Number')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        <div id="mobile_country_code" class="phone_country_code">+1</div>\n        <input id="mobile" name="mobile" class="phone" type="text" value=""/>\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <h2>');
      __out.push(__sanitize(t('Payment Information')));
      __out.push('</h2>\n\n      <div class="form_clear"></div>\n\n      <span><span id="top_of_form" class="error_message"></span></span>\n\n\n      <div class="form_label">\n        <label for="card_number">');
      __out.push(__sanitize(t('Credit Card Number')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        <input id="card_number" name="card_number" type="text" value=""/>\n        <!--img id="card_icon" src="/web/img/cc_mastercard_24.png"-->\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <div class="form_label">\n        <label for="card_expiration_month">');
      __out.push(__sanitize(t('Expiration Date')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        <select id="card_expiration_month" name="card_expiration_month">\n          <option value="01">01</option>\n          <option value="02">02</option>\n          <option value="03">03</option>\n          <option value="04">04</option>\n          <option value="05">05</option>\n          <option value="06">06</option>\n          <option value="07">07</option>\n          <option value="08">08</option>\n          <option value="09">09</option>\n          <option value="10">10</option>\n          <option value="11">11</option>\n          <option value="12">12</option>\n        </select>\n\n        <select id="card_expiration_year" name="card_expiration_year">\n          <option value="2011">2011</option>\n          <option value="2012">2012</option>\n          <option value="2013">2013</option>\n          <option value="2014">2014</option>\n          <option value="2015">2015</option>\n          <option value="2016">2016</option>\n          <option value="2017">2017</option>\n          <option value="2018">2018</option>\n          <option value="2019">2019</option>\n          <option value="2020">2020</option>\n          <option value="2021">2021</option>\n        </select>\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <div class="form_label">\n        <label for="card_number">');
      __out.push(__sanitize(t('Security Code')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        <input id="card_code" name="card_code" type="text" value="" class="half" />\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <div class="form_label">\n        <label for="use_case">');
      __out.push(__sanitize(t('Type of Card')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        <select id="use_case" name="use_case">\n          <option value="personal">');
      __out.push(__sanitize(t('Personal')));
      __out.push('</option>\n          <option value="business">');
      __out.push(__sanitize(t('Business')));
      __out.push('</option>\n        </select>\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <h2>');
      __out.push(__sanitize(t('Promotion Code')));
      __out.push('</h2>\n\n      <div class="form_label">\n        <label for="promotion_code">');
      __out.push(__sanitize(t('Code')));
      __out.push('</label>\n      </div>\n      <div class="form_input">\n        <input id="promotion_code" name="promotion_code" type="text" value="');
      __out.push(__sanitize(this.invite));
      __out.push('">\n        <span class="error_message"></span>\n      </div>\n\n      <div class="form_clear"></div>\n\n      <h2>');
      __out.push(__sanitize(t('Legal Information')));
      __out.push('</h2>\n\n      <p>');
      __out.push(t('Sign Up Agreement', {
        terms_link: "<a href='https://www.uber.com/terms' target='_blank' style='line-height:11px;'>" + (t('Terms and Conditions')) + "</a>",
        privacy_link: "<a href='https://www.uber.com/privacy' target='_blank' style='line-height:11px;'>" + (t('Privacy Policy')) + "</a>"
      }));
      __out.push('</p>\n\n      <p>');
      __out.push(t('Message and Data Rates Disclosure', {
        help_string: "<strong>" + (t('HELP')) + "</strong>",
        stop_string: "<strong>" + (t('STOP')) + "</strong>"
      }));
      __out.push('</p>\n\n      <p style="display:none" id="terms_error" class="error_message">');
      __out.push(__sanitize(t('Sign Up Agreement Error')));
      __out.push('</p>\n\n      <div id="signup_terms">\n        <p>\n          <input type="checkbox" name="signup_terms_agreement" />\n          <label for="signup_terms_agreement"><strong>');
      __out.push(t('I Agree'));
      __out.push('</strong></label>\n        </p>\n      </div>\n\n      <div class="formSubmitButton"><button type="submit" class="button" data-theme="a" id="sign_up_submit_button"><span>');
      __out.push(__sanitize(t('Sign Up')));
      __out.push('</span></button></div>\n\n    </form>\n  </div>\n</div>\n\n<div id="small_container_shadow"></div>\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/clients/trip_detail": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      var distance, fareBreakdown, printFares, printStar, _ref, _ref2, _ref3, _ref4, _ref5, _ref6;
      var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };
      printStar = function() {
        return __out.push('\n  <img alt="Star" src="/web/img/star.png"/>\n');
      };
      __out.push('\n');
      fareBreakdown = this.trip.get('fare_breakdown');
      __out.push('\n\n');
      printFares = __bind(function(fare, index, list) {
        var _ref;
        __out.push('\n\n  <li>\n    <span class="fare">');
        __out.push(__sanitize(app.helpers.formatCurrency(fare['amount'], false, (_ref = this.trip.get('fare_breakdown_local')) != null ? _ref.currency : void 0)));
        __out.push('</span><br/>\n    <span class="subtext">');
        __out.push(__sanitize(fare['name']));
        __out.push('</span>\n    ');
        if (fare['variable_rate'] !== 0) {
          __out.push('\n      <br><span class="subtext">');
          __out.push(__sanitize("" + (app.helpers.formatCurrency(fare['variable_rate'], false, this.trip.get('fare_breakdown_local'))) + " x " + (app.helpers.roundNumber(fare['input_amount'], 3)) + " " + fare['input_type']));
          __out.push('</span>\n    ');
        }
        __out.push('\n  </li>\n  ');
        if (index !== list.length - 1) {
          __out.push('\n    <li class="math">+</li>\n  ');
        }
        return __out.push('\n');
      }, this);
      __out.push('\n\n');
      __out.push(require('templates/clients/modules/sub_header').call(this, {
        heading: t("Trip Details")
      }));
      __out.push('\n\n\n<div id="main_content">\n  <div class="clear"></div>\n  <div>\n    <div id="trip_details_map"></div>\n    <div id="trip_details_info">\n      <h2>\n          ');
      __out.push(__sanitize(t('Your Trip')));
      __out.push('\n      </h2>\n\n      <div id="avatars">\n        <h3>');
      __out.push(__sanitize(t('Driver')));
      __out.push('</h3>\n        <img alt="Driver image" height="45" src="');
      __out.push(__sanitize((_ref = this.trip.get('driver')) != null ? _ref.picture_url : void 0));
      __out.push('" width="45" />\n        <span>');
      __out.push(__sanitize((_ref2 = this.trip.get('driver')) != null ? _ref2.first_name : void 0));
      __out.push('</span>\n\n        <div class="clear"></div>\n      </div>\n\n        <h3>');
      __out.push(__sanitize(t('Rating')));
      __out.push('</h3>\n          ');
      _(this.trip.get('driver_rating')).times(printStar);
      __out.push('\n        <h3>');
      __out.push(__sanitize(t('Trip Info')));
      __out.push('</h3>\n        <table>\n          <tr class="first">\n            <td class="label">');
      __out.push(__sanitize(t('Pickup time:')));
      __out.push('</td>\n            <td>');
      __out.push(__sanitize(app.helpers.formatDate(this.trip.get('begintrip_at'), true, this.trip.get('city').timezone)));
      __out.push('</td>\n          </tr>\n          <tr>\n            <td class="label">');
      __out.push(__sanitize(t("" + (app.helpers.capitaliseFirstLetter((_ref3 = this.trip.get('city')) != null ? (_ref4 = _ref3.country) != null ? _ref4.distance_unit : void 0 : void 0)) + "s:")));
      __out.push('</td>\n            ');
      distance = this.trip.get('distance', 0);
      __out.push('\n            ');
      if (((_ref5 = this.trip.get('city')) != null ? (_ref6 = _ref5.country) != null ? _ref6.distance_unit : void 0 : void 0) === "kilometer") {
        __out.push('\n            ');
        distance = distance * 1.609344;
        __out.push('\n            ');
      }
      __out.push('\n            <td>');
      __out.push(__sanitize(app.helpers.roundNumber(distance, 2)));
      __out.push('</td>\n          </tr>\n          <tr>\n            <td class="label">');
      __out.push(__sanitize(t('Trip time:')));
      __out.push('</td>\n            <td>');
      __out.push(__sanitize(app.helpers.formatSeconds(this.trip.get('duration'))));
      __out.push('</td>\n          </tr>\n          <tr>\n            <td class="label">');
      __out.push(__sanitize(t('Fare:')));
      __out.push('</td>\n            <td>');
      __out.push(__sanitize(app.helpers.formatTripFare(this.trip)));
      __out.push('</td>\n          </tr>\n        </table>\n\n        <p><button class="resendReceipt"><span>Resend Receipt</span></button> <span class="resendReceiptSuccess success"></span><span class="resendReceiptError error"></span></p>\n\n        <p><a id="fare_review" href="">');
      __out.push(__sanitize(t('Request a fare review')));
      __out.push('</a></p>\n    </div>\n    <div class="clear"></div>\n\n    <div id="fare_review_box">\n\n      <span class="success_message" style="display:none;">');
      __out.push(__sanitize(t("Fare Review Submitted")));
      __out.push('</span>\n      <div id="fare_review_form_wrapper">\n        <p>');
      __out.push(__sanitize(t("Fair Price Consideration")));
      __out.push('</p>\n        <div id="pricing_breakdown">\n         <h3>');
      __out.push(__sanitize(t('Your Fare Calculation')));
      __out.push('</h3>\n\n          <h4>');
      __out.push(__sanitize(t('Charges')));
      __out.push('</h4>\n          <ul>\n            ');
      _.each(fareBreakdown['charges'], printFares);
      __out.push('\n            <div class="clear"></div>\n          </ul>\n\n          <h4>');
      __out.push(__sanitize(t('Discounts')));
      __out.push('</h4>\n          <ul>\n            ');
      _.each(fareBreakdown['discounts'], printFares);
      __out.push('\n            <div class="clear"></div>\n          </ul>\n\n          <h4>');
      __out.push(__sanitize(t('Total Charge')));
      __out.push('</h4>\n          <ul>\n            <li class="math">=</li>\n            <li class="valign"><span>$');
      __out.push(__sanitize(this.trip.get('fare')));
      __out.push('</span></li>\n            <div class="clear"></div>\n          </ul>\n        </div>\n        <ul>\n          <li>');
      __out.push(t('Uber Pricing Information Message', {
        learn_link: "<a href='" + (app.config.get('url')) + "/learn'>" + (t('Uber pricing information')) + "</a>"
      }));
      __out.push('</li>\n          <li>');
      __out.push(__sanitize(t('GPS Point Capture Disclosure')));
      __out.push('</li>\n        </ul>\n\n        <p>');
      __out.push(__sanitize(t('Fare Review Note')));
      __out.push('</p>\n        <span class="error_message" style="display:none;">');
      __out.push(__sanitize(t('Fare Review Error')));
      __out.push('</span>\n        <form id="form_review_form" action="" method="">\n          <input type="hidden" id="tripid" name="tripid" value="');
      __out.push(__sanitize(this.trip.get('id')));
      __out.push('">\n          <textarea id="form_review_message" name="message"></textarea>\n          <div class="clear"></div>\n          <button id="submit_fare_review" type="submit" class="button" data-theme="a"><span>');
      __out.push(__sanitize(t('Submit')));
      __out.push('</span></button>\n        </form>\n        <button class="button" id="fare_review_hide" data-theme="a"><span>');
      __out.push(__sanitize(t('Cancel')));
      __out.push('</span></button>\n      </div>\n    </div>\n    <div class="clear"></div>\n  </div>\n</div>\n<div id="main_shadow"></div>\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "templates/shared/menu": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      __out.push('<div id="menu_main">\n  <div class="logo">\n    <a href="/"><img src="/web/img/logo-charcoal.png"></a>\n  </div>\n  <div class="nav">\n    <ul>\n      ');
      if (this.type === 'guest') {
        __out.push('\n        <li><a class="" href="/#!/sign-up" id="">');
        __out.push(__sanitize(t("Sign Up")));
        __out.push('</a></li>\n        <li><a class="" href="https://www.uber.com/learn" id="">');
        __out.push(__sanitize(t("Learn More")));
        __out.push('</a></li>\n        <li><a class="" href="http://blog.uber.com" id="">');
        __out.push(__sanitize(t("Blog")));
        __out.push('</a></li>\n        <li><a class="" href="/#!/sign-in">');
        __out.push(__sanitize(t("Sign In")));
        __out.push(' &raquo;</a></li>\n      ');
      }
      __out.push('\n      ');
      if (this.type === 'client') {
        __out.push('\n        ');
        if ($.cookie('user') && JSON.parse($.cookie('user')).is_admin) {
          __out.push('\n            <li><a class="" href="/#!/request" id="">');
          __out.push(__sanitize(t("Ride Request")));
          __out.push('</a></li>\n        ');
        }
        __out.push('\n        <li><a class="" href="/#!/dashboard" id="">');
        __out.push(__sanitize(t("Dashboard")));
        __out.push('</a></li>\n        <li><a class="" href="/#!/invite" id="">');
        __out.push(__sanitize(t("Invite Friends")));
        __out.push('</a></li>\n        <li><a class="" href="/#!/promotions" id="">');
        __out.push(__sanitize(t("Promotions")));
        __out.push('</a></li>\n        <li><a class="" href="/#!/billing" id="">');
        __out.push(__sanitize(t("Billing")));
        __out.push('</a></li>\n        <li><a class="" href="/#!/settings/information" id="">');
        __out.push(__sanitize(t("Settings")));
        __out.push('</a></li>\n        <li><a class="" href="/#!/sign-out">');
        __out.push(__sanitize(t("Sign Out")));
        __out.push(' &raquo;</a></li>\n      ');
      }
      __out.push('\n    </ul>\n  </div>\n</div>\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "translations/en": function(exports, require, module) {(function() {
  exports.translations = {
    "Uber": "Uber",
    "Sign Up": "Sign Up",
    "Ride Request": "Ride Request",
    "Invite Friends": "Invite Friends",
    "Promotions": "Promotions",
    "Billing": "Billing",
    "Settings": "Settings",
    "Forgot Password?": "Forgot Password?",
    "Password Recovery": "Password Recovery",
    "Login": "Login",
    "Trip Detail": "Trip Detail",
    "Password Reset": "Password Reset",
    "Confirm Email": "Confirm Email",
    "Request Ride": "Request Ride",
    "Credit Card Number": "Credit Card Number",
    "month": "month",
    "01-Jan": "01-Jan",
    "02-Feb": "02-Feb",
    "03-Mar": "03-Mar",
    "04-Apr": "04-Apr",
    "05-May": "05-May",
    "06-Jun": "06-Jun",
    "07-Jul": "07-Jul",
    "08-Aug": "08-Aug",
    "09-Sep": "09-Sep",
    "10-Oct": "10-Oct",
    "11-Nov": "11-Nov",
    "12-Dec": "12-Dec",
    "year": "year",
    "CVV": "CVV",
    "Category": "Category",
    "personal": "personal",
    "business": "business",
    "Default Credit Card": "Default Credit Card",
    "Add Credit Card": "Add Credit Card",
    "Expiry": "Expiry",
    "default card": "default card",
    "make default": "make default",
    "Edit": "Edit",
    "Delete": "Delete",
    "Expiry Month": "Expiry Month",
    "Expiry Year": "Expiry Year",
    "Unable to Verify Card": "Unable to verify card at this time. Please try again later.",
    "Credit Card Update Succeeded": "Your card has been successfully updated!",
    "Credit Card Update Failed": "We couldn't save your changes. Please try again in a few minutes.",
    "Credit Card Delete Succeeded": "Your card has been deleted!",
    "Credit Card Delete Failed": "We were unable to delete your card. Please try again later.",
    "Credit Card Update Category Succeeded": "Successfully changed card category!",
    "Credit Card Update Category Failed": "We couldn't change your card category. Please try again in a few minutes.",
    "Credit Card Update Default Succeeded": "Successfully changed default card!",
    "Credit Card Update Default Failed": "We couldn't change your default card. Please try again in a few minutes.",
    "Hello Greeting": "Hello, <%= name %>",
    "Card Ending in": "Card Ending in",
    "Trip Map": "Trip Map",
    "Amount": "Amount: <%= amount %>",
    "Last Attempt to Bill": "Last Attempt to Bill: <%= date %>",
    "Charge": "Charge",
    "Uber Credit Balance Note": "Your account has an UberCredit balance of <%= amount %>. When billing for trips, we'll deplete your UberCredit balance before applying charges to your credit card.",
    "Please Add Credit Card": "Please add a credit card to bill your outstanding charges.",
    "Credit Cards": "Credit Cards",
    "add a new credit card": "add a new credit card",
    "Account Balance": "Account Balance",
    "Arrears": "Arrears",
    "Billing Succeeded": "Your card was successfully billed.",
    "Confirm Email Succeeded": "Successfully confirmed email token, redirecting to log in page...",
    "Confirm Email Failed": "Unable to confirm email. Please contact support@uber.com if this problem persists.",
    "Email Already Confirmed": "Your email address has already been confirmed, redirecting to log in page...",
    "Credit Card Added": "Credit Card Added",
    "No Credit Card": "No Credit Card",
    "Mobile Number Confirmed": "Mobile Number Confirmed",
    "No Confirmed Mobile": "No Confirmed Mobile",
    "E-mail Address Confirmed": "E-mail Address Confirmed",
    "No Confirmed E-mail": "No Confirmed E-mail",
    'Reply to sign up text': 'Reply "GO" to the text message you received at sign up.',
    "Resend text message": "Resend text message",
    "Click sign up link": "Click the link in the email you received at sign up.",
    "Resend email": "Resend email",
    "Add a credit card to ride": "Add a credit card and you'll be ready to ride Uber.",
    "Your Most Recent Trip": "Your Most Recent Trip",
    "details": "details",
    "Your Trip History ": "Your Trip History ",
    "Status": "Status",
    "Here's how it works:": "Here's how it works:",
    "Show all trips": "Show all trips",
    "Set your location:": "Set your location:",
    "App search for address": "iPhone/Android app: fix the pin or search for an address",
    "SMS text address": "SMS: text your address to UBRCAB (827222)",
    "Confirm pickup request": "Confirm your pickup request",
    "Uber sends ETA": "Uber will send you an ETA (usually within 5-10 minutes)",
    "Car arrives": "When your car is arriving, Uber will inform you again.",
    "Ride to destination": "Hop in the car and tell the driver your destination.",
    "Thank your driver": "That’s it! Please thank your driver but remember that your tip is included and no cash is necessary.",
    "Trip started here": "Trip started here",
    "Trip ended here": "Trip ended here",
    "Sending Email": "Sending email...",
    "Resend Email Succeeded": "We just sent the email. Please click on the confirmation link you recieve.",
    "Resend Email Failed": "There was an error sending the email. Please contact support if the problem persists.",
    "Resend Text Succeeded": 'We just sent the text message. Please reply "GO" to the message you recieve. It may take a few minutes for the message to reach you phone.',
    "Resend Text Failed": "There was an error sending the text message. Please contact support if the problem persists.",
    "Password Reset Error": "There was an error processing your password reset request.",
    "New Password": "New Password",
    "Forgot Password": "Forgot Password",
    "Forgot Password Error": "Your email address could not be found. Please make sure to use the same email address you used when you signed up.",
    "Forgot Password Success": "Please check your email for a link to reset your password.",
    "Forgot Password Enter Email": 'Enter your email address and Uber will send you a link to reset your password. If you remember your password, you can <a href="/#!/sign-in">sign in here</a>.',
    "Invite friends": "Invite friends",
    "Give $ Get $": "Give $10, Get $10",
    "Give $ Get $ Description": "Every friend you invite to Uber gets $10 of Uber credit. After someone you’ve invited takes his/her first ride, you get $10 of Uber credits too!",
    "What are you waiting for?": "So, what are you waiting for? Invite away!",
    "Tweet": "Tweet",
    "Invite Link": "Email or IM this link to your friends:",
    "Email Address": "Email Address",
    "Reset Password": "Reset Password",
    "Enter Promotion Code": "If you have a promotion code, enter it here:",
    "Your Active Promotions": "Your Active Promotions",
    "Code": "Code",
    "Details": "Details",
    "Trips Remaining": "Trips Remaining",
    "Expires": "Expires",
    "No Active Promotions": "There are no active promotions on your account.",
    "Your Available Promotions": "Your Available Promotions",
    "Where do you want us to pick you up?": "Where do you want us to pick you up?",
    "Address to search": "Address to search",
    "Search": "Search",
    "Driver Name:": "Driver Name:",
    "Driver #:": "Driver #:",
    "Pickup Address:": "Pickup Address:",
    "Add to Favorite Locations": "Add to Favorite Locations",
    "Star": "Star",
    "Nickname:": "Nickname:",
    "Add": "Add",
    "Your last trip": "Your last trip",
    "Please rate your driver:": "Please rate your driver:",
    "Comments: (optional)": "Comments: (optional)",
    "Rate Trip": "Rate Trip",
    "Pickup time:": "Pickup time:",
    "Miles:": "Miles:",
    "Trip time:": "Trip time:",
    "Fare:": "Fare:",
    "Favorite Locations": "Favorite Locations",
    "Search Results": "Search Results",
    "You have no favorite locations saved.": "You have no favorite locations saved.",
    "Loading...": "Loading...",
    "Request Pickup": "Request Pickup",
    "Cancel Pickup": "Cancel Pickup",
    "Requesting Closest Driver": "Requesting the closest driver to pick you up...",
    "En Route": "You are currently en route...",
    "Rate Last Trip": "Please rate your trip to make another request",
    "Rate Before Submitting": "Please rate your trip before submitting the form",
    "Address too short": "Address too short",
    "or did you mean": "or did you mean",
    "Search Address Failed": "Unable to find the given address. Please enter another address close to your location.",
    "Sending pickup request...": "Sending pickup request...",
    "Cancel Request Prompt": "Are you sure you want to cancel your request?",
    "Cancel Request Arrived Prompt": 'Are you sure you want to cancel your request? Your driver has arrived so there is a $10 cancellation fee. It may help to call your driver now',
    "Favorite Location Nickname Length Error": "Nickname has to be atleast 3 characters",
    "Favorite Location Save Succeeded": "Location Saved!",
    "Favorite Location Save Failed": "Unable to save your location. Please try again later.",
    "Favorite Location Title": "Favorite Location <%= id %>",
    "Search Location Title": "Search Location <%= id %>",
    "ETA Message": "ETA: Around <%= minutes %> Minutes",
    "Nearest Cab Message": "The closest driver is approximately <%= minutes %> minute(s) away",
    "Arrival ETA Message": "Your Uber will arrive in about <%= minutes %> minute(s)",
    "Arriving Now Message": "Your Uber is arriving now...",
    "Rating Driver Failed": "Unable to contact server. Please try again later or email support if this issue persists.",
    "Account Information": "Account Information",
    "Mobile Phone Information": "Mobile Phone Information",
    "settings": "settings",
    "Information": "Information",
    "Picture": "Picture",
    "Change password": "Change password",
    "Your current Picture": "Your current Picture",
    "Your Favorite Locations": "Your Favorite Locations",
    "You have no favorite locations saved.": "You have no favorite locations saved.",
    "Purpose of Mobile": "We send text messages to your mobile phone to tell you when your driver is arriving. You can also request trips using text messages.",
    "Country": "Country",
    "Mobile Number": "Mobile Number",
    "Submit": "Submit",
    "Favorite Location": "Favorite Location",
    "No Approximate Address": "Could not find an approximate address",
    "Address:": "Address:",
    "Information Update Succeeded": "Your information has been updated!",
    "Information Update Failed": "We couldn't update your information. Please try again in few minutes or contact support if the problem persists.",
    "Location Delete Succeeded": "Location deleted!",
    "Location Delete Failed": "We were unable to delete your favorite location. Please try again later or contact support of the issue persists.",
    "Location Edit Succeeded": "Changes Saved!",
    "Location Edit Failed": "We couldn't save your changes. Please try again in a few minutes.",
    "Picture Update Succeeded": "Your picture has been updated!",
    "Picture Update Failed": "We couldn't change your picture. Please try again in a few minutes.",
    "Personal Information": "Personal Information",
    "Mobile Phone Number": "Mobile Phone Number",
    "Payment Information": "Payment Information",
    "Purpose of Credit Card": "We keep your credit card on file so that your trip go as fast as possible. You will not be charged until you take a trip.",
    "Your card will not be charged until you take a trip.": "Your card will not be charged until you take a trip.",
    "Credit Card Number": "Credit Card Number",
    "Expiration Date": "Expiration Date",
    "Promotion Code": "Promotion Code",
    "Enter Promo Here": "If you have a code for a promotion, invitation or group deal, you can enter it here.",
    "Promotion Code Input Label": "Promotion, Invite or Groupon Code (optional)",
    "Terms and Conditions": "Terms and Conditions",
    "HELP": "HELP",
    "STOP": "STOP",
    "Legal Information": "Legal Information",
    "Sign Up Agreement": "By signing up, I agree to the Uber <%= terms_link %> and <%= privacy_link %> and understand that Uber is a request tool, not a transportation carrier.",
    "Sign Up Agreement Error": "You must agree to the Uber Terms and Conditions and Privacy Policy to continue.",
    "Message and Data Rates Disclosure": "Message and Data Rates May Apply. Reply <%= help_string %> to 827-222 for help. Reply <%= stop_string %> to 827-222 to stop texts. For additional assistance, visit support.uber.com or call (866) 576-1039. Supported Carriers: AT&amp;T, Sprint, Verizon, and T-Mobile.",
    "I Agree": "I agree to the Terms &amp; Conditions and Privacy Policy",
    "Security Code": "Security Code",
    "Type of Card": "Type of Card",
    "Personal": "Personal",
    "Business": "Business",
    "Code": "Code",
    "Zip or Postal Code": "Zip or Postal Code",
    "Your Trip": "Your Trip",
    "Trip Info": "Trip Info",
    "Request a fare review": "Request a fare review",
    "Fare Review Submitted": "Your fare review has been submitted. We'll get back to you soon about your request. Sorry for any inconvenience this may have caused!",
    "Fair Price Consideration": "We're committed to delivering Uber service at a fair price. Before requesting a fare review, please consider:",
    "Your Fare Calculation": "Your Fare Calculation",
    "Charges": "Charges",
    "Discounts": "Discounts",
    "Total Charge": "Total Charge",
    "Uber pricing information": "Uber pricing information",
    "Uber Pricing Information Message": "<%= learn_link %> is published on our website.",
    "GPS Point Capture Disclosure": "Due to a finite number of GPS point captures, corners on your trip map may appear cut off or rounded. These minor inaccuracies result in a shorter measured distance, which always results in a cheaper trip.",
    "Fare Review Note": "Please elaborate on why this trip requires a fare review. Your comments below will help us better establish the correct price for your trip:",
    "Fare Review Error": "There was an error submitting the review. Please ensure that you have a message.",
    "Sign In": "Sign In"
  };
}).call(this);
}, "translations/fr": function(exports, require, module) {(function() {
  exports.translations = {
    "Uber": "Uber",
    "Sign Up": "Inscription",
    "Ride Request": "Passer une Commande",
    "Invite Friends": "Inviter vos Amis",
    "Promotions": "Promotions",
    "Billing": "Paiement",
    "Settings": "Paramètres",
    "Forgot Password?": "Mot de passe oublié ?",
    "Password Recovery": "Récupération du mot de passe",
    "Login": "Connexion",
    "Trip Detail": "Détail de la Course",
    "Password Reset": "Réinitialisation du mot de passe",
    "Confirm Email": "Confirmation de l’e-mail",
    "Request Ride": "Passer une Commande",
    "Credit Card Number": "Numéro de Carte de Crédit",
    "month": "mois",
    "01-Jan": "01-Jan",
    "02-Feb": "02-Fév",
    "03-Mar": "03-Mar",
    "04-Apr": "04-Avr",
    "05-May": "05-Mai",
    "06-Jun": "06-Juin",
    "07-Jul": "07-Jui",
    "08-Aug": "08-Aoû",
    "09-Sep": "09-Sep",
    "10-Oct": "10-Oct",
    "11-Nov": "11-Nov",
    "12-Dec": "12-Déc",
    "year": "année",
    "CVV": "Code de Sécurité",
    "Category": "Type",
    "personal": "personnel",
    "business": "entreprise",
    "Default Credit Card": "Carte par Défaut",
    "Add Credit Card": "Ajouter une Carte",
    "Expiry": "Expire",
    "default card": "carte par défaut",
    "make default": "choisir par défaut",
    "Edit": "Modifier",
    "Delete": "Supprimer",
    "Expiry Month": "Mois d’Expiration",
    "Expiry Year": "Année d’Expiration",
    "Unable to Verify Card": "Impossible de vérifier la carte pour le moment. Merci de réessayer un peu plus tard.",
    "Credit Card Update Succeeded": "Votre carte a été mise à jour avec succès !",
    "Credit Card Update Failed": "Nous ne pouvons enregistrer vos changements. Merci de réessayer dans quelques minutes.",
    "Credit Card Delete Succeeded": "Votre carte a été supprimée !",
    "Credit Card Delete Failed": "Nous n’avons pas été en mesure de supprimer votre carte. Merci de réessayer plus tard.",
    "Credit Card Update Category Succeeded": "Changement de catégorie de carte réussi !",
    "Credit Card Update Category Failed": "Nous ne pouvons pas changer la catégorie de votre carte. Merci de réessayer dans quelques minutes.",
    "Credit Card Update Default Succeeded": "Carte par défaut changée avec succès !",
    "Credit Card Update Default Failed": "Nous ne pouvons pas changer votre carte par défaut. Merci de réessayer dans quelques minutes.",
    "Hello Greeting": "Bonjour, <%= name %>",
    "Card Ending in": "La carte expire dans",
    "Trip Map": "Carte des Courses",
    "Amount": "Montant: <%= amount %>",
    "Last Attempt to Bill": "Dernière tentative de prélèvement : <%= date %>",
    "Charge": "Débit",
    "Uber Credit Balance Note": "Votre compte a un solde de <%= amount %> UberCredits. Lorsque nous facturons des courses, nous réduirons votre solde d’UberCredits avant de prélever votre carte de crédit.",
    "Please Add Credit Card": "Merci d’ajouter une carte de crédit pour que nous puissions vous facturer.",
    "Credit Cards": "Cartes de crédit",
    "add a new credit card": "Ajouter une nouvelle carte de crédit",
    "Account Balance": "Solde du compte",
    "Arrears": "Arriérés",
    "Billing Succeeded": "Votre carte a été correctement débitée.",
    "Confirm Email Succeeded": "L’adresse e-mail a bien été validée, vous êtes redirigé vers le tableau de bord...",
    "Confirm Email Failed": "Impossible de confirmer l’adresse e-mail. Merci de contacter support@uber.com si le problème persiste.",
    "Credit Card Added": "Carte de crédit ajoutée",
    "No Credit Card": "Pas de carte de crédit",
    "Mobile Number Confirmed": "Numéro de téléphone confirmé",
    "No Confirmed Mobile": "Pas de numéro de téléphone confirmé",
    "E-mail Address Confirmed": "Adresse e-mail confirmée",
    "No Confirmed E-mail": "Pas d’adresse e-mail confirmée",
    'Reply to sign up text': 'Répondre "GO" au SMS que vous avez reçu à l’inscription.',
    "Resend text message": "Renvoyer le SMS",
    "Click sign up link": "Cliquez sur le lien contenu dans l’e-mail reçu à l’inscription.",
    "Resend email": "Renvoyer l’e-mail",
    "Add a credit card to ride": "Ajouter une carte de crédit et vous serez prêt à voyager avec Uber.",
    "Your Most Recent Trip": "Votre course la plus récente",
    "details": "détails",
    "Your Trip History": "Historique de votre trajet",
    "Status": "Statut",
    "Here's how it works:": "Voici comment ça marche :",
    "Show all trips": "Montrer toutes les courses",
    "Set your location:": "Définir votre position :",
    "App search for address": "Application iPhone/Android : positionner la punaise ou rechercher une adresse",
    "SMS text address": "SMS : envoyez votre adresse à UBRCAB (827222)",
    "Confirm pickup request": "Validez la commande",
    "Uber sends ETA": "Uber envoie un temps d’attente estimé (habituellement entre 5 et 10 minutes)",
    "Car arrives": "Lorsque votre voiture arrive, Uber vous en informera encore..",
    "Ride to destination": "Montez dans la voiture et donnez votre destination au chauffeur.",
    "Thank your driver": "C’est tout ! Remerciez le chauffeur mais souvenez-vous que les pourboires sont compris et qu’il n’est pas nécessaire d’avoir du liquide sur soi.",
    "Trip started here": "La course a commencé ici.",
    "Trip ended here": "La course s’est terminée ici.",
    "Sending Email": "Envoi de l’e-mail...",
    "Resend Email Succeeded": "Nous venons d’envoyer l’e-mail. Merci de cliquer sur le lien de confirmation que vous avez reçu.",
    "Resend Email Failed": "Il y a eu un problème lors de l’envoi de l’email. Merci de contacter le support si le problème persiste.",
    "Resend Text Succeeded": 'Nous venons d’envoyer le SMS. Merci de répondre "GO" au message que vous avez reçu. Il se peut que cela prenne quelques minutes pour que le message arrive sur votre téléphone.',
    "Resend Text Failed": "Il y a eu un problème lors de l’envoi du SMS. Merci de contacter le support si le problème persiste.",
    "Password Reset Error": "Il y a eu une error lors de la réinitialisation de votre mot de passe.",
    "New Password:": "Nouveau mot de passe:",
    "Forgot Password Error": "Votre nom d’utilisateur / adresse email ne peut être trouvé. Merci d’utiliser la même qu’à l’inscription.",
    "Forgot Password Success": "Merci de consulter votre boîte mail pour suivre la demande de ‘réinitialisation de mot de passe.",
    "Forgot Password Enter Email": "Merci de saisir votre adresse email et nous vous enverrons un lien vous permettant de réinitialiser votre mot de passe :",
    "Invite friends": "Inviter vos amis",
    "Give $ Get $": "Donnez $10, Recevez $10",
    "Give $ Get $ Description": "Chaque ami que vous invitez à Uber  recevra $10 de crédits Uber. Dès lors qu’une personne que vous aurez invité aura utilisé Uber pour la première, vous recevrez $10 de crédits Uber également !",
    "What are you waiting for?": "N’attendez plus ! Lancez les invitations !",
    "Tweet": "Tweeter",
    "Invite Link": "Envoyez ce lien par email ou messagerie instantanée à vos amis :",
    "Enter Promotion Code": "Si vous avez un code promo, saisissez-le ici:",
    "Your Active Promotions": "Vos Codes Promos Actifs",
    "Code": "Code",
    "Details": "Détails",
    "Trips Remaining": "Courses restantes",
    "Expires": "Expire",
    "No Active Promotions": "Vous n’avez pas de code promo actif.",
    "Your Available Promotions": "Votres Promos Disponibles",
    "Where do you want us to pick you up?": "Où souhaitez-vous que nous vous prenions en charge ?",
    "Address to search": "Adresse à rechercher",
    "Search": "Chercher",
    "Driver Name:": "Nom du chauffeur:",
    "Driver #:": "# Chauffeur:",
    "Pickup Address:": "Lieu de prise en charge:",
    "Add to Favorite Locations": "Ajoutez aux Lieux Favoris",
    "Star": "Étoiles",
    "Nickname:": "Pseudo",
    "Add": "Ajouter",
    "Your last trip": "Votre dernière course",
    "Please rate your driver:": "Merci de noter votre chauffeur :",
    "Comments: (optional)": "Commentaires: (optionnel)",
    "Rate Trip": "Notez votre course",
    "Pickup time:": "Heure de Prise en Charge :",
    "Miles:": "Kilomètres :",
    "Trip time:": "Temps de course :",
    "Fare:": "Tarif :",
    "Favorite Locations": "Lieux Favoris",
    "Search Results": "Résultats",
    "You have no favorite locations saved.": "Vous n’avez pas de lieux de prise en charge favoris.",
    "Loading...": "Chargement...",
    "Request Pickup": "Commander ici",
    "Cancel Pickup": "Annuler",
    "Requesting Closest Driver": "Nous demandons au chauffeur le plus proche de vous prendre en charge...",
    "En Route": "Vous êtes actuellement en route...",
    "Rate Last Trip": "Merci de noter votre précédent trajet pour faire une autre course.",
    "Rate Before Submitting": "Merci de noter votre trajet avant de le valider.",
    "Address too short": "L’adresse est trop courte",
    "or did you mean": "ou vouliez-vous dire",
    "Search Address Failed": "Impossible de trouver l’adresse spécifiée. Merci de saisir une autre adresse proche de l’endroit où vous vous trouvez.",
    "Sending pickup request...": "Envoi de la demande de prise en charge...",
    "Cancel Request Prompt": "Voulez-vous vraiment annuler votre demande ?",
    "Cancel Request Arrived Prompt": 'Voulez-vous vraiment annuler votre demande ? Votre chauffeur est arrivé, vous serez donc facturé de $10 de frais d’annulation. Il pourrait être utile que vous appeliez votre chauffeur maintenant.',
    "Favorite Location Nickname Length Error": "Le pseudo doit faire au moins 3 caractères de long",
    "Favorite Location Save Succeeded": "Adresse enregistrée !",
    "Favorite Location Save Failed": "Impossible d’enregistrer votre adresse. Merci de réessayer ultérieurement.",
    "Favorite Location Title": "Adresse favorie <%= id %>",
    "Search Location Title": "Recherche d’adresse <%= id %>",
    "ETA Message": "Temps d’attente estimé: environ <%= minutes %> minutes",
    "Nearest Cab Message": "Le chauffeur le plus proche sera là dans <%= minutes %> minute(s)",
    "Arrival ETA Message": "Votre chauffeur arrivera dans <%= minutes %> minute(s)",
    "Arriving Now Message": "Votre chauffeur est en approche...",
    "Rating Driver Failed": "Impossible de contacter le serveur. Merci de réessayer ultérieurement ou de contacter le support si le problème persiste.",
    "settings": "Paramètres",
    "Information": "Information",
    "Picture": "Photo",
    "Change password": "Modifier votre mot de passe",
    "Your current Picture": "Votre photo",
    "Your Favorite Locations": "Vos lieux favoris",
    "You have no favorite locations saved.": "Vous n’avez pas de lieu favori",
    "Account Information": "Informations Personnelles",
    "Mobile Phone Information": "Informations de Mobile",
    "Change Your Password": "Changez votre mot de passe.",
    "Country": "Pays",
    "Language": "Langue",
    "Favorite Location": "Lieu favori",
    "No Approximate Address": "Impossible de trouver une adresse même approximative",
    "Address:": "Adresse :",
    "Information Update Succeeded": "Vos informations ont été mises à jour !",
    "Information Update Failed": "Nous n’avons pas pu mettre à jour vos informations. Merci de réessayer dans quelques instants ou de contacter le support si le problème persiste.",
    "Location Delete Succeeded": "Adresse supprimée !",
    "Location Delete Failed": "Nous n’avons pas pu supprimée votre adresse favorie. Merci de réessayer plus tard ou de contacter le support si le problème persiste.",
    "Location Edit Succeeded": "Modifications sauvegardées !",
    "Location Edit Failed": "Nous n’avons pas pu sauvegarder vos modifications. Merci de réessayer dans quelques minutes.",
    "Picture Update Succeeded": "Votre photo a été mise à jour !",
    "Picture Update Failed": "Nous n’avons pas pu mettre à jour votre photo. Merci de réessayer dans quelques instants.",
    "Personal Information": "Informations Personnelles",
    "Mobile Phone Number": "Numéro de Téléphone Portable",
    "Payment Information": "Informations de Facturation",
    "Your card will not be charged until you take a trip.": "Votre carte ne sera pas débitée avant votre premier trajet.",
    "Card Number": "Numéro de Carte",
    "Promotion Code Input Label": "Code promo, code d’invitation ou “deal” acheté en ligne (optionnel)",
    "Terms and Conditions": "Conditions Générales",
    "HELP": "HELP",
    "STOP": "STOP",
    "Sign Up Agreement": "En souscrivant, j’accepte les <%= terms_link %> et <%= privacy_link %>  et comprends qu’Uber est un outil de commande de chauffeur, et non un transporteur.",
    "Sign Up Agreement Error": "Vous devez accepter les Conditions Générales d’utilisation d’Uber Terms and Conditions et la Politique de Confidentialité pour continuer.",
    "Message and Data Rates Disclosure": "Les frais d’envoi de SMS et de consommation de données peuvent s’appliquer. Répondez <%= help_string %> au 827-222 pour obtenir de l’aide. Répondez <%= stop_string %> au 827-222 pour ne plus recevoir de SMS. Pour plus d’aide, visitez support.uber.com ou appelez le (866) 576-1039. Opérateurs supportés: AT&amp;T, Sprint, Verizon, T-Mobile, Orange, SFR et Bouygues Telecom.",
    "Zip/Postal Code": "Code Postal",
    "Expiration Date": "Date D'expiration",
    "Security Code": "Code de Sécurité",
    "Type of Card": "Type",
    "Personal": "Personnel",
    "Business": "Entreprise",
    "Promotion Code": "Code Promo",
    "Legal Information": "Mentions Légales",
    "I Agree": "J'accepte.",
    "Your Trip": "Votre Course",
    "Trip Info": "Informations de la Course",
    "Request a fare review": "Demander un contrôle du tarif",
    "Fare Review Submitted": "Votre demande de contrôle du tarif a été soumis. Nous reviendrons vers vous rapidement concernant cette demande. Nous nous excusons pour les dérangements éventuellement occasionnés !",
    "Fair Price Consideration": "Nous nous engageons à proposer Uber à un tarif juste. Avant de demander un contrôle du tarif, merci de prendre en compte :",
    "Your Fare Calculation": "Calcul du Prix",
    "Charges": "Coûts",
    "Discounts": "Réductions",
    "Total Charge": "Coût total",
    "Uber pricing information": "Information sur les prix d’Uber",
    "Uber Pricing Information Message": "<%= learn_link %> est disponible sur notre site web.",
    "GPS Point Capture Disclosure": "A cause d’un nombre limité de coordonnées GPS sauvegardées, les angles de votre trajet sur la carte peuvent apparaître coupés ou arrondis. Ces légères incohérences débouchent sur des distances mesurées plus courtes, ce qui implique toujours un prix du trajet moins élevé.",
    "Fare Review Note": "Merci de nous expliquer pourquoi le tarif de cette course nécessite d’être contrôlé. Vos commentaires ci-dessous nous aideront à établir un prix plus juste si nécessaire :",
    "Fare Review Error": "Il y a eu une erreur lors de l’envoi de la demande. Assurez-vous d’avoir bien ajouté une description à votre demande."
  };
}).call(this);
}, "views/clients/billing": function(exports, require, module) {(function() {
  var clientsBillingTemplate;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  }, __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };
  clientsBillingTemplate = require('templates/clients/billing');
  exports.ClientsBillingView = (function() {
    __extends(ClientsBillingView, UberView);
    function ClientsBillingView() {
      ClientsBillingView.__super__.constructor.apply(this, arguments);
    }
    ClientsBillingView.prototype.id = 'billing_view';
    ClientsBillingView.prototype.className = 'view_container';
    ClientsBillingView.prototype.events = {
      'click a#add_card': 'addCard',
      'click .charge_arrear': 'chargeArrear'
    };
    ClientsBillingView.prototype.render = function() {
      this.RefreshUserInfo(__bind(function() {
        var cards, newForm;
        this.HideSpinner();
        $(this.el).html(clientsBillingTemplate());
        if (USER.payment_gateway.payment_profiles.length === 0) {
          newForm = new app.views.clients.modules.creditcard;
          $(this.el).find("#add_card_wrapper").html(newForm.render(0).el);
        } else {
          cards = new app.views.clients.modules.creditcard;
          $("#cards").html(cards.render("all").el);
        }
        return this.delegateEvents();
      }, this));
      return this;
    };
    ClientsBillingView.prototype.addCard = function(e) {
      var newCard;
      e.preventDefault();
      newCard = new app.views.clients.modules.creditcard;
      $('#cards').append(newCard.render("new").el);
      return $("a#add_card").hide();
    };
    ClientsBillingView.prototype.chargeArrear = function(e) {
      var $el, arrearId, attrs, cardId, options, tryCharge;
      e.preventDefault();
      $(".error_message").text("");
      $el = $(e.currentTarget);
      arrearId = $el.attr('id');
      cardId = $el.parent().find('#card_to_charge').val();
      this.ShowSpinner('submit');
      tryCharge = new app.models.clientbills({
        id: arrearId
      });
      attrs = {
        payment_profile_id: cardId,
        dataType: 'json'
      };
      options = {
        success: __bind(function(data, textStatus, jqXHR) {
          $el.parent().find(".success_message").text(t("Billing Succeeded"));
          $el.hide();
          return $el.parent().find('#card_to_charge').hide();
        }, this),
        error: __bind(function(jqXHR, status, errorThrown) {
          return $el.parent().find(".error_message").text(JSON.parse(status.responseText).error);
        }, this),
        complete: __bind(function() {
          return this.HideSpinner();
        }, this)
      };
      return tryCharge.save(attrs, options);
    };
    return ClientsBillingView;
  })();
}).call(this);
}, "views/clients/confirm_email": function(exports, require, module) {(function() {
  var clientsConfirmEmailTemplate;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  }, __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };
  clientsConfirmEmailTemplate = require('templates/clients/confirm_email');
  exports.ClientsConfirmEmailView = (function() {
    __extends(ClientsConfirmEmailView, UberView);
    function ClientsConfirmEmailView() {
      ClientsConfirmEmailView.__super__.constructor.apply(this, arguments);
    }
    ClientsConfirmEmailView.prototype.id = 'confirm_email_view';
    ClientsConfirmEmailView.prototype.className = 'view_container';
    ClientsConfirmEmailView.prototype.render = function(token) {
      var attrs;
      $(this.el).html(clientsConfirmEmailTemplate());
      attrs = {
        data: {
          email_token: token
        },
        success: __bind(function(data, textStatus, jqXHR) {
          var show_dashboard;
          this.HideSpinner();
          show_dashboard = function() {
            return app.routers.clients.navigate('!/dashboard', true);
          };
          if (data.status === 'OK') {
            $('.success_message').show();
            return _.delay(show_dashboard, 3000);
          } else if (data.status === 'ALREADY_COMFIRMED') {
            $('.already_confirmed_message').show();
            return _.delay(show_dashboard, 3000);
          } else {
            return $('.error_message').show();
          }
        }, this),
        error: __bind(function(e) {
          this.HideSpinner();
          return $('.error_message').show();
        }, this),
        complete: function(status) {
          return $('#attempt_text').hide();
        },
        dataType: 'json',
        type: 'PUT',
        url: "" + API + "/users/self"
      };
      $.ajax(attrs);
      this.ShowSpinner('submit');
      return this;
    };
    return ClientsConfirmEmailView;
  })();
}).call(this);
}, "views/clients/dashboard": function(exports, require, module) {(function() {
  var clientsDashboardTemplate;
  var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; }, __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  clientsDashboardTemplate = require('templates/clients/dashboard');
  exports.ClientsDashboardView = (function() {
    var displayFirstTrip;
    __extends(ClientsDashboardView, UberView);
    function ClientsDashboardView() {
      this.showAllTrips = __bind(this.showAllTrips, this);
      this.render = __bind(this.render, this);
      ClientsDashboardView.__super__.constructor.apply(this, arguments);
    }
    ClientsDashboardView.prototype.id = 'dashboard_view';
    ClientsDashboardView.prototype.className = 'view_container';
    ClientsDashboardView.prototype.events = {
      'click a.confirmation': 'confirmationClick',
      'click #resend_email': 'resendEmail',
      'click #resend_mobile': 'resendMobile',
      'click #show_all_trips': 'showAllTrips'
    };
    ClientsDashboardView.prototype.render = function() {
      var displayPage, downloadTrips;
      this.HideSpinner();
      displayPage = __bind(function() {
        $(this.el).html(clientsDashboardTemplate());
        this.confirmationsSetup();
        return this.RequireMaps(__bind(function() {
          if (USER.trips.models[0]) {
            if (!USER.trips.models[0].get("points")) {
              return USER.trips.models[0].fetch({
                data: {
                  relationships: 'points'
                },
                success: __bind(function() {
                  this.CacheData("USERtrips", USER.trips);
                  return displayFirstTrip();
                }, this)
              });
            } else {
              return displayFirstTrip();
            }
          }
        }, this));
      }, this);
      downloadTrips = __bind(function() {
        return this.DownloadUserTrips(displayPage, false, 10);
      }, this);
      this.RefreshUserInfo(downloadTrips);
      return this;
    };
    displayFirstTrip = __bind(function() {
      var bounds, endPos, map, myOptions, path, polyline, startPos;
      myOptions = {
        zoom: 12,
        mapTypeId: google.maps.MapTypeId.ROADMAP,
        zoomControl: false,
        rotateControl: false,
        panControl: false,
        mapTypeControl: false,
        scrollwheel: false
      };
      if (USER.trips.length === 10) {
        $("#show_all_trips").show();
      }
      if (USER.trips.length > 0) {
        map = new google.maps.Map(document.getElementById("trip_details_map"), myOptions);
        bounds = new google.maps.LatLngBounds();
        path = [];
        _.each(USER.trips.models[0].get('points'), __bind(function(point) {
          path.push(new google.maps.LatLng(point.lat, point.lng));
          return bounds.extend(_.last(path));
        }, this));
        map.fitBounds(bounds);
        startPos = new google.maps.Marker({
          position: _.first(path),
          map: map,
          title: t('Trip started here'),
          icon: 'https://uber-static.s3.amazonaws.com/marker_start.png'
        });
        endPos = new google.maps.Marker({
          position: _.last(path),
          map: map,
          title: t('Trip ended here'),
          icon: 'https://uber-static.s3.amazonaws.com/marker_end.png'
        });
        polyline = new google.maps.Polyline({
          path: path,
          strokeColor: '#003F87',
          strokeOpacity: 1,
          strokeWeight: 5
        });
        return polyline.setMap(map);
      }
    }, ClientsDashboardView);
    ClientsDashboardView.prototype.confirmationsSetup = function() {
      var blink, cardForm, element, _ref, _ref2, _ref3, _ref4, _ref5;
      blink = function(element) {
        var opacity;
        opacity = 0.5;
        if (element.css('opacity') === "0.5") {
          opacity = 1.0;
        }
        return element.fadeTo(2000, opacity, function() {
          return blink(element);
        });
      };
      if (((_ref = window.USER) != null ? (_ref2 = _ref.payment_gateway) != null ? (_ref3 = _ref2.payment_profiles) != null ? _ref3.length : void 0 : void 0 : void 0) === 0) {
        element = $('#confirmed_credit_card');
        cardForm = new app.views.clients.modules.creditcard;
        $('#card.info').append(cardForm.render().el);
        blink(element);
      }
      if (((_ref4 = window.USER) != null ? _ref4.confirm_email : void 0) === false) {
        element = $('#confirmed_email');
        blink(element);
      }
      if ((((_ref5 = window.USER) != null ? _ref5.confirm_mobile : void 0) != null) === false) {
        element = $('#confirmed_mobile');
        return blink(element);
      }
    };
    ClientsDashboardView.prototype.confirmationClick = function(e) {
      e.preventDefault();
      $('.info').hide();
      $('#more_info').show();
      switch (e.currentTarget.id) {
        case "card":
          return $('#card.info').slideToggle();
        case "mobile":
          return $('#mobile.info').slideToggle();
        case "email":
          return $('#email.info').slideToggle();
      }
    };
    ClientsDashboardView.prototype.resendEmail = function(e) {
      var $el;
      e.preventDefault();
      $el = $(e.currentTarget);
      $el.removeAttr('href').prop({
        disabled: true
      });
      $el.html(t("Sending Email"));
      return $.ajax({
        type: 'GET',
        url: API + '/users/request_confirm_email',
        data: {
          token: USER.token
        },
        dataType: 'json',
        success: __bind(function(data, textStatus, jqXHR) {
          return $el.html(t("Resend Email Succeeded"));
        }, this),
        error: __bind(function(jqXHR, textStatus, errorThrown) {
          return $el.html(t("Resend Email Failed"));
        }, this)
      });
    };
    ClientsDashboardView.prototype.resendMobile = function(e) {
      var $el;
      e.preventDefault();
      $el = $(e.currentTarget);
      $el.removeAttr('href').prop({
        disabled: true
      });
      $el.html("Sending message...");
      return $.ajax({
        type: 'GET',
        url: API + '/users/request_confirm_mobile',
        data: {
          token: USER.token
        },
        dataType: 'json',
        success: __bind(function(data, textStatus, jqXHR) {
          return $el.html(t("Resend Text Succeeded"));
        }, this),
        error: __bind(function(jqXHR, textStatus, errorThrown) {
          return $el.html(t("Resend Text Failed"));
        }, this)
      });
    };
    ClientsDashboardView.prototype.showAllTrips = function(e) {
      e.preventDefault();
      $(e.currentTarget).hide();
      return this.DownloadUserTrips(this.render, true, 1000);
    };
    return ClientsDashboardView;
  }).call(this);
}).call(this);
}, "views/clients/forgot_password": function(exports, require, module) {(function() {
  var clientsForgotPasswordTemplate;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  }, __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };
  clientsForgotPasswordTemplate = require('templates/clients/forgot_password');
  exports.ClientsForgotPasswordView = (function() {
    __extends(ClientsForgotPasswordView, UberView);
    function ClientsForgotPasswordView() {
      ClientsForgotPasswordView.__super__.constructor.apply(this, arguments);
    }
    ClientsForgotPasswordView.prototype.id = 'forgotpassword_view';
    ClientsForgotPasswordView.prototype.className = 'view_container modal_view_container';
    ClientsForgotPasswordView.prototype.events = {
      "submit #password_reset": "passwordReset",
      "click #password_reset_submit": "passwordReset",
      "submit #forgot_password": "forgotPassword",
      "click #forgot_password_submit": "forgotPassword"
    };
    ClientsForgotPasswordView.prototype.render = function(token) {
      this.HideSpinner();
      $(this.el).html(clientsForgotPasswordTemplate({
        token: token
      }));
      this.delegateEvents();
      return this;
    };
    ClientsForgotPasswordView.prototype.forgotPassword = function(e) {
      var attrs;
      e.preventDefault();
      $('.success_message').hide();
      $(".error_message").hide();
      attrs = {
        data: {
          login: $("#login").val()
        },
        success: __bind(function(data, textStatus, jqXHR) {
          this.HideSpinner();
          $('.success_message').show();
          return $("#forgot_password").hide();
        }, this),
        error: __bind(function(e) {
          this.HideSpinner();
          return $('.error_message').show();
        }, this),
        dataType: 'json',
        type: 'PUT',
        url: "" + API + "/users/forgot_password"
      };
      $.ajax(attrs);
      return this.ShowSpinner('submit');
    };
    ClientsForgotPasswordView.prototype.passwordReset = function(e) {
      var attrs;
      e.preventDefault();
      attrs = {
        data: {
          email_token: $("#token").val(),
          password: $("#password").val()
        },
        success: __bind(function(data, textStatus, jqXHR) {
          this.HideSpinner();
          $.cookie('token', data.token);
          amplify.store('USERjson', data);
          app.refreshMenu();
          return location.hash = '!/dashboard';
        }, this),
        error: __bind(function(e) {
          this.HideSpinner();
          return $('#error_reset').show();
        }, this),
        dataType: 'json',
        type: 'PUT',
        url: "" + API + "/users/self"
      };
      $.ajax(attrs);
      return this.ShowSpinner('submit');
    };
    return ClientsForgotPasswordView;
  })();
}).call(this);
}, "views/clients/invite": function(exports, require, module) {(function() {
  var clientsInviteTemplate;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  clientsInviteTemplate = require('templates/clients/invite');
  exports.ClientsInviteView = (function() {
    __extends(ClientsInviteView, UberView);
    function ClientsInviteView() {
      ClientsInviteView.__super__.constructor.apply(this, arguments);
    }
    ClientsInviteView.prototype.id = 'invite_view';
    ClientsInviteView.prototype.className = 'view_container';
    ClientsInviteView.prototype.render = function() {
      this.ReadUserInfo();
      this.HideSpinner();
      $(this.el).html(clientsInviteTemplate());
      console.log(screen);
      return this;
    };
    return ClientsInviteView;
  })();
}).call(this);
}, "views/clients/login": function(exports, require, module) {(function() {
  var clientsLoginTemplate;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  clientsLoginTemplate = require('templates/clients/login');
  exports.ClientsLoginView = (function() {
    __extends(ClientsLoginView, UberView);
    function ClientsLoginView() {
      ClientsLoginView.__super__.constructor.apply(this, arguments);
    }
    ClientsLoginView.prototype.id = 'login_view';
    ClientsLoginView.prototype.className = 'view_container modal_view_container';
    ClientsLoginView.prototype.events = {
      'submit form': 'authenticate',
      'click button': 'authenticate'
    };
    ClientsLoginView.prototype.initialize = function() {
      _.bindAll(this, 'render');
      return this.render();
    };
    ClientsLoginView.prototype.render = function() {
      this.HideSpinner();
      $(this.el).html(clientsLoginTemplate());
      this.delegateEvents();
      return this.place();
    };
    ClientsLoginView.prototype.authenticate = function(e) {
      e.preventDefault();
      return $.ajax({
        type: 'POST',
        url: API + '/auth/web_login/client',
        data: {
          login: $("#login").val(),
          password: $("#password").val()
        },
        dataType: 'json',
        success: function(data, textStatus, jqXHR) {
          $.cookie('user', JSON.stringify(data));
          $.cookie('token', data.token);
          amplify.store('USERjson', data);
          $('header').html(app.views.shared.menu.render().el);
          return app.routers.clients.navigate('!/dashboard', true);
        },
        error: function(jqXHR, textStatus, errorThrown) {
          $.cookie('user', null);
          $.cookie('token', null);
          if (jqXHR.status === 403) {
            $.cookie('redirected_user', JSON.stringify(JSON.parse(jqXHR.responseText).error_obj), {
              domain: '.uber.com'
            });
            window.location = 'http://partners.uber.com/';
          }
          return $('.error_message').html(JSON.parse(jqXHR.responseText).error).hide().fadeIn();
        }
      });
    };
    return ClientsLoginView;
  })();
}).call(this);
}, "views/clients/modules/credit_card": function(exports, require, module) {(function() {
  var creditCardTemplate;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  }, __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };
  creditCardTemplate = require('templates/clients/modules/credit_card');
  exports.CreditCardView = (function() {
    __extends(CreditCardView, UberView);
    function CreditCardView() {
      CreditCardView.__super__.constructor.apply(this, arguments);
    }
    CreditCardView.prototype.id = 'creditcard_view';
    CreditCardView.prototype.className = 'module_container';
    CreditCardView.prototype.events = {
      'submit #credit_card_form': 'processNewCard',
      'click #new_card': 'processNewCard',
      'change #card_number': 'showCardType',
      'click .edit_card_show': 'showEditCard',
      'click .edit_card': 'editCard',
      'click .delete_card': 'deleteCard',
      'click .make_default': 'makeDefault',
      'change .use_case': 'saveUseCase'
    };
    CreditCardView.prototype.initialize = function() {
      return app.collections.paymentprofiles.bind("refresh", __bind(function() {
        return this.RefreshUserInfo(__bind(function() {
          this.render("all");
          return this.HideSpinner();
        }, this));
      }, this));
    };
    CreditCardView.prototype.render = function(cards) {
      if (cards == null) {
        cards = "new";
      }
      if (cards === "all") {
        app.collections.paymentprofiles.reset(USER.payment_gateway.payment_profiles);
        cards = app.collections.paymentprofiles;
      }
      $(this.el).html(creditCardTemplate({
        cards: cards
      }));
      return this;
    };
    CreditCardView.prototype.processNewCard = function(e) {
      var $el, attrs, model, options;
      e.preventDefault();
      this.ClearGlobalStatus();
      $el = $("#credit_card_form");
      $el.find('.error_message').html("");
      attrs = {
        card_number: $el.find('#card_number').val(),
        card_code: $el.find('#card_code').val(),
        card_expiration_month: $el.find('#card_expiration_month').val(),
        card_expiration_year: $el.find('#card_expiration_year').val(),
        use_case: $el.find('#use_case').val(),
        "default": $el.find('#default_check').prop("checked")
      };
      options = {
        statusCode: {
          200: __bind(function(e) {
            this.HideSpinner();
            $('#cc_form_wrapper').hide();
            app.collections.paymentprofiles.trigger("refresh");
            $(this.el).remove();
            $("a#add_card").show();
            return $('section').html(app.views.clients.billing.render().el);
          }, this),
          406: __bind(function(e) {
            var error, errors, _i, _len, _ref, _results;
            this.HideSpinner();
            errors = JSON.parse(e.responseText);
            _ref = _.keys(errors);
            _results = [];
            for (_i = 0, _len = _ref.length; _i < _len; _i++) {
              error = _ref[_i];
              _results.push(error === "top_of_form" ? $("#top_of_form").html(errors[error]) : $("#credit_card_form").find("#" + error).parent().find(".error_message").html(errors[error]));
            }
            return _results;
          }, this),
          420: __bind(function(e) {
            this.HideSpinner();
            return $("#top_of_form").html(t("Unable to Verify Card"));
          }, this)
        }
      };
      this.ShowSpinner("submit");
      model = new app.models.paymentprofile;
      model.save(attrs, options);
      return app.collections.paymentprofiles.add(model);
    };
    CreditCardView.prototype.showCardType = function(e) {
      var $el, reAmerica, reDiscover, reMaster, reVisa, validCard;
      reVisa = /^4\d{3}-?\d{4}-?\d{4}-?\d{4}$/;
      reMaster = /^5[1-5]\d{2}-?\d{4}-?\d{4}-?\d{4}$/;
      reAmerica = /^6011-?\d{4}-?\d{4}-?\d{4}$/;
      reDiscover = /^3[4,7]\d{13}$/;
      $el = $("#card_logos");
      validCard = false;
      if (e.currentTarget.value.match(reVisa)) {
        validCard = true;
      } else if (e.currentTarget.value.match(reMaster)) {
        $el.css('background-position', "-60px");
        validCard = true;
      } else if (e.currentTarget.value.match(reAmerica)) {
        $el.css('background-position', "-120px");
        validCard = true;
      } else if (e.currentTarget.value.match(reDiscover)) {
        $el.css('background-position', "-180px");
        validCard = true;
      }
      if (validCard) {
        $el.css('width', "60px");
        return $el.css('margin-left', "180px");
      } else {
        $el.css('width', "250px");
        return $el.css('margin-left', "80px");
      }
    };
    CreditCardView.prototype.showEditCard = function(e) {
      var $el, id;
      e.preventDefault();
      $el = $(e.currentTarget);
      if ($el.html() === t("Edit")) {
        id = $el.html(t("Cancel")).parents("tr").attr("id").substring(1);
        return $("#e" + id).show();
      } else {
        id = $el.html(t("Edit")).parents("tr").attr("id").substring(1);
        return $("#e" + id).hide();
      }
    };
    CreditCardView.prototype.editCard = function(e) {
      var $el, attrs, id, options;
      e.preventDefault();
      this.ClearGlobalStatus();
      $el = $(e.currentTarget).parents("td");
      id = $el.parents("tr").attr("id").substring(1);
      $el.attr('disabled', 'disabled');
      this.ShowSpinner('submit');
      attrs = {
        card_expiration_month: $el.find('#card_expiration_month').val(),
        card_expiration_year: $el.find('#card_expiration_year').val(),
        card_code: $el.find('#card_code').val()
      };
      options = {
        success: __bind(function(response) {
          this.HideSpinner();
          this.ShowSuccess(t("Credit Card Update Succeeded"));
          $("#e" + id).hide();
          $("#d" + id).find(".edit_card_show").html(t("Edit"));
          return app.collections.paymentprofiles.trigger("refresh");
        }, this),
        error: __bind(function(e) {
          this.HideSpinner();
          this.ShowError(t("Credit Card Update Failed"));
          return $el.removeAttr('disabled');
        }, this)
      };
      app.collections.paymentprofiles.models[id].set(attrs);
      return app.collections.paymentprofiles.models[id].save({}, options);
    };
    CreditCardView.prototype.deleteCard = function(e) {
      var $el, id, options;
      e.preventDefault();
      $el = $(e.currentTarget).parents("td");
      id = $el.parents("tr").attr("id").substring(1);
      this.ClearGlobalStatus();
      this.ShowSpinner('submit');
      options = {
        success: __bind(function(response) {
          this.ShowSuccess(t("Credit Card Delete Succeeded"));
          $("form").hide();
          app.collections.paymentprofiles.trigger("refresh");
          return $('section').html(app.views.clients.billing.render().el);
        }, this),
        error: __bind(function(xhr, e) {
          this.HideSpinner();
          return this.ShowError(t("Credit Card Delete Failed"));
        }, this)
      };
      return app.collections.paymentprofiles.models[id].destroy(options);
    };
    CreditCardView.prototype.saveUseCase = function(e) {
      var $el, attrs, id, options, use_case;
      this.ClearGlobalStatus();
      $el = $(e.currentTarget);
      use_case = $el.val();
      id = $el.parents("tr").attr("id").substring(1);
      attrs = {
        use_case: use_case
      };
      options = {
        success: __bind(function(response) {
          return this.ShowSuccess(t("Credit Card Update Category Succeeded"));
        }, this),
        error: __bind(function(e) {
          return this.ShowError(t("Credit Card Update Category Failed"));
        }, this)
      };
      app.collections.paymentprofiles.models[id].set(attrs);
      return app.collections.paymentprofiles.models[id].save({}, options);
    };
    CreditCardView.prototype.makeDefault = function(e) {
      var $el, attrs, id, options;
      e.preventDefault();
      this.ClearGlobalStatus();
      $el = $(e.currentTarget).parents("td");
      id = $el.parents("tr").attr("id").substring(1);
      attrs = {
        "default": true
      };
      options = {
        success: __bind(function(response) {
          this.ShowSuccess(t("Credit Card Update Default Succeeded"));
          return app.collections.paymentprofiles.trigger("refresh");
        }, this),
        error: __bind(function(e) {
          return this.ShowError(t("Credit Card Update Default Failed"));
        }, this)
      };
      app.collections.paymentprofiles.models[id].set(attrs);
      return app.collections.paymentprofiles.models[id].save({}, options);
    };
    return CreditCardView;
  })();
}).call(this);
}, "views/clients/promotions": function(exports, require, module) {(function() {
  var clientsPromotionsTemplate;
  var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; }, __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  clientsPromotionsTemplate = require('templates/clients/promotions');
  exports.ClientsPromotionsView = (function() {
    __extends(ClientsPromotionsView, UberView);
    function ClientsPromotionsView() {
      this.render = __bind(this.render, this);
      ClientsPromotionsView.__super__.constructor.apply(this, arguments);
    }
    ClientsPromotionsView.prototype.id = 'promotions_view';
    ClientsPromotionsView.prototype.className = 'view_container';
    ClientsPromotionsView.prototype.events = {
      'submit form': 'submitPromo',
      'click button': 'submitPromo'
    };
    ClientsPromotionsView.prototype.initialize = function() {
      if (this.model) {
        return this.RefreshUserInfo(this.render);
      }
    };
    ClientsPromotionsView.prototype.render = function() {
      var renderTemplate;
      this.ReadUserInfo();
      renderTemplate = __bind(function() {
        $(this.el).html(clientsPromotionsTemplate({
          promos: window.USER.unexpired_client_promotions || []
        }));
        return this.HideSpinner();
      }, this);
      this.DownloadUserPromotions(renderTemplate);
      return this;
    };
    ClientsPromotionsView.prototype.submitPromo = function(e) {
      var attrs, model, options, refreshTable;
      e.preventDefault();
      this.ClearGlobalStatus();
      refreshTable = __bind(function() {
        $('section').html(this.render().el);
        return this.HideSpinner();
      }, this);
      attrs = {
        code: $('#code').val()
      };
      options = {
        success: __bind(function(response) {
          this.HideSpinner();
          if (response.get('first_name')) {
            return this.ShowSuccess("Your promotion has been applied in the form of an account credit. <a href='#!/billing'>Click here</a> to check your balance.");
          } else {
            this.ShowSuccess("Your promotion has successfully been applied");
            return this.RefreshUserInfo(this.render, true);
          }
        }, this),
        statusCode: {
          400: __bind(function(e) {
            this.ShowError(JSON.parse(e.responseText).error);
            return this.HideSpinner();
          }, this)
        }
      };
      this.ShowSpinner("submit");
      model = new app.models.promotions;
      return model.save(attrs, options);
    };
    return ClientsPromotionsView;
  })();
}).call(this);
}, "views/clients/request": function(exports, require, module) {(function() {
  var clientsRequestTemplate;
  var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; }, __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  clientsRequestTemplate = require('templates/clients/request');
  exports.ClientsRequestView = (function() {
    __extends(ClientsRequestView, UberView);
    function ClientsRequestView() {
      this.AjaxCall = __bind(this.AjaxCall, this);
      this.AskDispatch = __bind(this.AskDispatch, this);
      this.removeMarkers = __bind(this.removeMarkers, this);
      this.displaySearchLoc = __bind(this.displaySearchLoc, this);
      this.displayFavLoc = __bind(this.displayFavLoc, this);
      this.showFavLoc = __bind(this.showFavLoc, this);
      this.addToFavLoc = __bind(this.addToFavLoc, this);
      this.removeCabs = __bind(this.removeCabs, this);
      this.requestRide = __bind(this.requestRide, this);
      this.rateTrip = __bind(this.rateTrip, this);
      this.locationChange = __bind(this.locationChange, this);
      this.panToLocation = __bind(this.panToLocation, this);
      this.clickLocation = __bind(this.clickLocation, this);
      this.searchLocation = __bind(this.searchLocation, this);
      this.mouseoutLocation = __bind(this.mouseoutLocation, this);
      this.mouseoverLocation = __bind(this.mouseoverLocation, this);
      this.fetchTripDetails = __bind(this.fetchTripDetails, this);
      this.submitRating = __bind(this.submitRating, this);
      this.setStatus = __bind(this.setStatus, this);
      this.initialize = __bind(this.initialize, this);
      ClientsRequestView.__super__.constructor.apply(this, arguments);
    }
    ClientsRequestView.prototype.id = 'request_view';
    ClientsRequestView.prototype.className = 'view_container';
    ClientsRequestView.prototype.pollInterval = 2 * 1000;
    ClientsRequestView.prototype.events = {
      "submit #search_form": "searchAddress",
      "click .locations_link": "locationLinkHandle",
      "mouseover .location_row": "mouseoverLocation",
      "mouseout .location_row": "mouseoutLocation",
      "click .location_row": "clickLocation",
      "click #search_location": "searchLocation",
      "click #pickupHandle": "pickupHandle",
      "click .stars": "rateTrip",
      "submit #rating_form": "submitRating",
      "click #addToFavButton": "showFavLoc",
      "click #favLocNickname": "selectInputText",
      "submit #favLoc_form": "addToFavLoc"
    };
    ClientsRequestView.prototype.status = "";
    ClientsRequestView.prototype.pickupMarker = "https://uber-static.s3.amazonaws.com/pickup_marker.png";
    ClientsRequestView.prototype.cabMarker = "https://uber-static.s3.amazonaws.com/cab_marker.png";
    ClientsRequestView.prototype.initialize = function() {
      var displayCabs;
      displayCabs = __bind(function() {
        return this.AskDispatch("NearestCab");
      }, this);
      this.showCabs = _.throttle(displayCabs, this.pollInterval);
      return this.numSearchToDisplay = 1;
    };
    ClientsRequestView.prototype.setStatus = function(status) {
      var autocomplete;
      if (this.status === status) {
        return;
      }
      try {
        google.maps.event.trigger(this.map, 'resize');
      } catch (_e) {}
      switch (status) {
        case "init":
          this.AskDispatch("StatusClient");
          this.status = "init";
          return this.ShowSpinner("load");
        case "ready":
          this.HideSpinner();
          $(".panel").hide();
          $("#top_bar").fadeIn();
          $("#location_panel").fadeIn();
          $("#location_panel_control").fadeIn();
          $("#pickupHandle").attr("class", "button_green").fadeIn().find("span").html(t("Request Pickup"));
          this.pickup_icon.setDraggable(true);
          this.map.panTo(this.pickup_icon.getPosition());
          this.showCabs();
          try {
            this.pickup_icon.setMap(this.map);
            this.displayFavLoc();
            autocomplete = new google.maps.places.Autocomplete(document.getElementById('address'), {
              types: ['geocode']
            });
            autocomplete.bindTo('bounds', this.map);
          } catch (_e) {}
          return this.status = "ready";
        case "searching":
          this.HideSpinner();
          this.removeMarkers();
          $(".panel").hide();
          $("#top_bar").fadeOut();
          $("#status_message").html(t("Requesting Closest Driver"));
          $("#pickupHandle").attr("class", "button_red").fadeIn().find("span").html(t("Cancel Pickup"));
          this.pickup_icon.setDraggable(false);
          this.pickup_icon.setMap(this.map);
          return this.status = "searching";
        case "waiting":
          this.HideSpinner();
          this.removeMarkers();
          $(".panel").hide();
          $("#top_bar").fadeOut();
          $("#pickupHandle").attr("class", "button_red").fadeIn().find("span").html(t("Cancel Pickup"));
          $("#waiting_riding").fadeIn();
          this.pickup_icon.setDraggable(false);
          this.pickup_icon.setMap(this.map);
          return this.status = "waiting";
        case "arriving":
          this.HideSpinner();
          this.removeMarkers();
          $(".panel").hide();
          $("#top_bar").fadeOut();
          $("#pickupHandle").attr("class", "button_red").fadeIn().find("span").html(t("Cancel Pickup"));
          $("#waiting_riding").fadeIn();
          this.pickup_icon.setDraggable(false);
          this.pickup_icon.setMap(this.map);
          return this.status = "arriving";
        case "riding":
          this.HideSpinner();
          this.removeMarkers();
          $(".panel").hide();
          $("#top_bar").fadeOut();
          $("#pickupHandle").fadeIn().attr("class", "button_red").find("span").html(t("Cancel Pickup"));
          $("#waiting_riding").fadeIn();
          this.pickup_icon.setDraggable(false);
          this.status = "riding";
          return $("#status_message").html(t("En Route"));
        case "rate":
          this.HideSpinner();
          $(".panel").hide();
          $("#pickupHandle").fadeOut();
          $("#trip_completed_panel").fadeIn();
          $('#status_message').html(t("Rate Last Trip"));
          return this.status = "rate";
      }
    };
    ClientsRequestView.prototype.render = function() {
      this.ReadUserInfo();
      this.HideSpinner();
      this.ShowSpinner("load");
      $(this.el).html(clientsRequestTemplate());
      this.cabs = [];
      this.RequireMaps(__bind(function() {
        var center, myOptions, streetViewPano;
        center = new google.maps.LatLng(37.7749295, -122.4194155);
        this.markers = [];
        this.pickup_icon = new google.maps.Marker({
          position: center,
          draggable: true,
          clickable: true,
          icon: this.pickupMarker
        });
        this.geocoder = new google.maps.Geocoder();
        myOptions = {
          zoom: 12,
          center: center,
          mapTypeId: google.maps.MapTypeId.ROADMAP,
          rotateControl: false,
          rotateControl: false,
          panControl: false
        };
        this.map = new google.maps.Map($(this.el).find("#map_wrapper_right")[0], myOptions);
        if (this.status === "ready") {
          this.pickup_icon.setMap(this.map);
        }
        if (geo_position_js.init()) {
          geo_position_js.getCurrentPosition(__bind(function(data) {
            var location;
            location = new google.maps.LatLng(data.coords.latitude, data.coords.longitude);
            this.pickup_icon.setPosition(location);
            this.map.panTo(location);
            return this.map.setZoom(16);
          }, this));
        }
        this.setStatus("init");
        streetViewPano = this.map.getStreetView();
        google.maps.event.addListener(streetViewPano, 'visible_changed', __bind(function() {
          if (streetViewPano.getVisible()) {
            this.pickupMarker = "https://uber-static.s3.amazonaws.com/pickup_marker_large.png";
            this.cabMarker = "https://uber-static.s3.amazonaws.com/cab_marker_large.png";
          } else {
            this.pickupMarker = "https://uber-static.s3.amazonaws.com/pickup_marker.png";
            this.cabMarker = "https://uber-static.s3.amazonaws.com/cab_marker.png";
          }
          this.pickup_icon.setIcon(this.pickupMarker);
          return _.each(this.cabs, __bind(function(cab) {
            return cab.setIcon(this.cabMarker);
          }, this));
        }, this));
        if (this.status === "ready") {
          return this.displayFavLoc();
        }
      }, this));
      return this;
    };
    ClientsRequestView.prototype.submitRating = function(e) {
      var $el, message, rating;
      e.preventDefault();
      $el = $(e.currentTarget);
      rating = 0;
      _(5).times(function(num) {
        if ($el.find(".stars#" + (num + 1)).attr("src") === "/web/img/star_active.png") {
          return rating = num + 1;
        }
      });
      if (rating === 0) {
        $("#status_message").html("").html(t("Rate Before Submitting"));
      } else {
        this.ShowSpinner("submit");
        this.AskDispatch("RatingDriver", {
          rating: rating
        });
      }
      message = $el.find("#comments").val().toString();
      if (message.length > 5) {
        return this.AskDispatch("Feedback", {
          message: message
        });
      }
    };
    ClientsRequestView.prototype.fetchTripDetails = function(id) {
      var trip;
      trip = new app.models.trip({
        id: id
      });
      return trip.fetch({
        data: {
          relationships: 'points,driver,city'
        },
        dataType: 'json',
        success: __bind(function() {
          var bounds, endPos, path, polyline, startPos;
          bounds = new google.maps.LatLngBounds();
          path = [];
          _.each(trip.get('points'), __bind(function(point) {
            path.push(new google.maps.LatLng(point.lat, point.lng));
            return bounds.extend(_.last(path));
          }, this));
          startPos = new google.maps.Marker({
            position: _.first(path),
            map: this.map,
            title: t("Trip started here"),
            icon: 'https://uber-static.s3.amazonaws.com/carstart.png'
          });
          endPos = new google.maps.Marker({
            position: _.last(path),
            map: this.map,
            title: t("Trip ended here"),
            icon: 'https://uber-static.s3.amazonaws.com/carstop.png'
          });
          polyline = new google.maps.Polyline({
            path: path,
            strokeColor: '#003F87',
            strokeOpacity: 1,
            strokeWeight: 5
          });
          polyline.setMap(this.map);
          this.map.fitBounds(bounds);
          $("#tripTime").html(app.helpers.parseDateTime(trip.get('pickup_local_time'), trip.get('city.timezone')));
          $("#tripDist").html(app.helpers.RoundNumber(trip.get('distance'), 2));
          $("#tripDur").html(app.helpers.FormatSeconds(trip.get('duration')));
          return $("#tripFare").html(app.helpers.FormatCurrency(trip.get('fare')));
        }, this)
      });
    };
    ClientsRequestView.prototype.searchAddress = function(e) {
      var $locationsDiv, address, alphabet, bounds, showResults;
      alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      try {
        e.preventDefault();
      } catch (_e) {}
      $('.error_message').html("");
      $locationsDiv = $("<table></table>");
      address = $('#address').val();
      bounds = new google.maps.LatLngBounds();
      if (address.length < 5) {
        $('#status_message').html(t("Address too short")).fadeIn();
        return false;
      }
      showResults = __bind(function(address, index) {
        var first_cell, row, second_cell;
        if (index < this.numSearchToDisplay) {
          first_cell = "<td class='marker_logo'><img src='https://www.google.com/mapfiles/marker" + alphabet[index] + ".png' /></td>";
          second_cell = "<td class='location_nickname_wrapper'>" + address.formatted_address + "</td>";
          row = $("<tr></tr>").attr("id", "s" + index).attr("class", "location_row").html(first_cell + second_cell);
          $locationsDiv.append(row);
        }
        if (index === this.numSearchToDisplay) {
          $locationsDiv.append("<tr><td colspan=2>" + (t('or did you mean')) + " </td></tr>");
          return $locationsDiv.append("<tr><td colspan=2><a id='search_location' href=''>" + address.formatted_address + "</a></td></tr>");
        }
      }, this);
      return this.geocoder.geocode({
        address: address
      }, __bind(function(result, status) {
        if (status !== "OK") {
          $('.error_message').html(t("Search Address Failed")).fadeIn();
          return;
        }
        _.each(result, showResults);
        $("#search_results").html($locationsDiv);
        this.locationChange("search");
        this.searchResults = result;
        return this.displaySearchLoc();
      }, this));
    };
    ClientsRequestView.prototype.mouseoverLocation = function(e) {
      var $el, id, marker;
      $el = $(e.currentTarget);
      id = $el.attr("id").substring(1);
      marker = this.markers[id];
      return marker.setAnimation(google.maps.Animation.BOUNCE);
    };
    ClientsRequestView.prototype.mouseoutLocation = function(e) {
      var $el, id, marker;
      $el = $(e.currentTarget);
      id = $el.attr("id").substring(1);
      marker = this.markers[id];
      return marker.setAnimation(null);
    };
    ClientsRequestView.prototype.searchLocation = function(e) {
      e.preventDefault();
      $("#address").val($(e.currentTarget).html());
      return this.searchAddress();
    };
    ClientsRequestView.prototype.favoriteClick = function(e) {
      var index, location;
      e.preventDefault();
      $(".favorites").attr("href", "");
      index = $(e.currentTarget).removeAttr("href").attr("id");
      location = new google.maps.LatLng(USER.locations[index].latitude, USER.locations[index].longitude);
      return this.panToLocation(location);
    };
    ClientsRequestView.prototype.clickLocation = function(e) {
      var id;
      id = $(e.currentTarget).attr("id").substring(1);
      return this.panToLocation(this.markers[id].getPosition());
    };
    ClientsRequestView.prototype.panToLocation = function(location) {
      this.map.panTo(location);
      this.map.setZoom(16);
      return this.pickup_icon.setPosition(location);
    };
    ClientsRequestView.prototype.locationLinkHandle = function(e) {
      var panelName;
      e.preventDefault();
      panelName = $(e.currentTarget).attr("id");
      return this.locationChange(panelName);
    };
    ClientsRequestView.prototype.locationChange = function(type) {
      $(".locations_link").attr("href", "").css("font-weight", "normal");
      switch (type) {
        case "favorite":
          $(".search_results").attr("href", "");
          $(".locations_link#favorite").removeAttr("href").css("font-weight", "bold");
          $("#search_results").hide();
          $("#favorite_results").fadeIn();
          return this.displayFavLoc();
        case "search":
          $(".favorites").attr("href", "");
          $(".locations_link#search").removeAttr("href").css("font-weight", "bold");
          $("#favorite_results").hide();
          $("#search_results").fadeIn();
          return this.displaySearchLoc();
      }
    };
    ClientsRequestView.prototype.rateTrip = function(e) {
      var rating;
      rating = $(e.currentTarget).attr("id");
      $(".stars").attr("src", "/web/img/star_inactive.png");
      return _(rating).times(function(index) {
        return $(".stars#" + (index + 1)).attr("src", "/web/img/star_active.png");
      });
    };
    ClientsRequestView.prototype.pickupHandle = function(e) {
      var $el, callback, message;
      e.preventDefault();
      $el = $(e.currentTarget).find("span");
      switch ($el.html()) {
        case t("Request Pickup"):
          _.delay(this.requestRide, 3000);
          $("#status_message").html(t("Sending pickup request..."));
          $el.html(t("Cancel Pickup")).parent().attr("class", "button_red");
          this.pickup_icon.setDraggable(false);
          this.map.panTo(this.pickup_icon.getPosition());
          return this.map.setZoom(18);
        case t("Cancel Pickup"):
          if (this.status === "ready") {
            $el.html(t("Request Pickup")).parent().attr("class", "button_green");
            return this.pickup_icon.setDraggable(true);
          } else {
            callback = __bind(function(v, m, f) {
              if (v) {
                this.AskDispatch("PickupCanceledClient");
                return this.setStatus("ready");
              }
            }, this);
            message = t("Cancel Request Prompt");
            if (this.status === "arriving") {
              message = 'Cancel Request Arrived Prompt';
            }
            return $.prompt(message, {
              buttons: {
                Ok: true,
                Cancel: false
              },
              callback: callback
            });
          }
      }
    };
    ClientsRequestView.prototype.requestRide = function() {
      if ($("#pickupHandle").find("span").html() === t("Cancel Pickup")) {
        this.AskDispatch("Pickup");
        return this.setStatus("searching");
      }
    };
    ClientsRequestView.prototype.removeCabs = function() {
      _.each(this.cabs, __bind(function(point) {
        return point.setMap(null);
      }, this));
      return this.cabs = [];
    };
    ClientsRequestView.prototype.addToFavLoc = function(e) {
      var $el, lat, lng, nickname;
      e.preventDefault();
      $el = $(e.currentTarget);
      $el.find(".error_message").html("");
      nickname = $el.find("#favLocNickname").val().toString();
      lat = $el.find("#pickupLat").val().toString();
      lng = $el.find("#pickupLng").val().toString();
      if (nickname.length < 3) {
        $el.find(".error_message").html(t("Favorite Location Nickname Length Error"));
        return;
      }
      this.ShowSpinner("submit");
      return $.ajax({
        type: 'POST',
        url: API + "/locations",
        dataType: 'json',
        data: {
          token: USER.token,
          nickname: nickname,
          latitude: lat,
          longitude: lng
        },
        success: __bind(function(data, textStatus, jqXHR) {
          return $el.html(t("Favorite Location Save Succeeded"));
        }, this),
        error: __bind(function(jqXHR, textStatus, errorThrown) {
          return $el.find(".error_message").html(t("Favorite Location Save Failed"));
        }, this),
        complete: __bind(function(data) {
          return this.HideSpinner();
        }, this)
      });
    };
    ClientsRequestView.prototype.showFavLoc = function(e) {
      $(e.currentTarget).fadeOut();
      return $("#favLoc_form").fadeIn();
    };
    ClientsRequestView.prototype.selectInputText = function(e) {
      e.currentTarget.focus();
      return e.currentTarget.select();
    };
    ClientsRequestView.prototype.displayFavLoc = function() {
      var alphabet, bounds;
      alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      this.removeMarkers();
      bounds = new google.maps.LatLngBounds();
      _.each(USER.locations, __bind(function(location, index) {
        var marker;
        marker = new google.maps.Marker({
          position: new google.maps.LatLng(location.latitude, location.longitude),
          map: this.map,
          title: t("Favorite Location Title", {
            id: alphabet != null ? alphabet[index] : void 0
          }),
          icon: "https://www.google.com/mapfiles/marker" + alphabet[index] + ".png"
        });
        this.markers.push(marker);
        bounds.extend(marker.getPosition());
        return google.maps.event.addListener(marker, 'click', __bind(function() {
          return this.pickup_icon.setPosition(marker.getPosition());
        }, this));
      }, this));
      this.pickup_icon.setPosition(_.first(this.markers).getPosition());
      return this.map.fitBounds(bounds);
    };
    ClientsRequestView.prototype.displaySearchLoc = function() {
      var alphabet;
      alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      this.removeMarkers();
      return _.each(this.searchResults, __bind(function(result, index) {
        var marker;
        if (index < this.numSearchToDisplay) {
          marker = new google.maps.Marker({
            position: result.geometry.location,
            map: this.map,
            title: t("Search Location Title", {
              id: alphabet != null ? alphabet[index] : void 0
            }),
            icon: "https://www.google.com/mapfiles/marker" + alphabet[index] + ".png"
          });
          this.markers.push(marker);
          return this.panToLocation(result.geometry.location);
        }
      }, this));
    };
    ClientsRequestView.prototype.removeMarkers = function() {
      _.each(this.markers, __bind(function(marker) {
        return marker.setMap(null);
      }, this));
      return this.markers = [];
    };
    ClientsRequestView.prototype.AskDispatch = function(ask, options) {
      var attrs, lowestETA, processData, showCab;
      if (ask == null) {
        ask = "";
      }
      if (options == null) {
        options = {};
      }
      switch (ask) {
        case "NearestCab":
          attrs = {
            latitude: this.pickup_icon.getPosition().lat(),
            longitude: this.pickup_icon.getPosition().lng()
          };
          lowestETA = 99999;
          showCab = __bind(function(cab) {
            var point;
            point = new google.maps.Marker({
              position: new google.maps.LatLng(cab.latitude, cab.longitude),
              map: this.map,
              icon: this.cabMarker,
              title: t("ETA Message", {
                minutes: app.helpers.FormatSeconds(cab != null ? cab.eta : void 0, true)
              })
            });
            if (cab.eta < lowestETA) {
              lowestETA = cab.eta;
            }
            return this.cabs.push(point);
          }, this);
          processData = __bind(function(data, textStatus, jqXHR) {
            if (this.status === "ready") {
              this.removeCabs();
              if (data.sorry) {
                $("#status_message").html(data.sorry).fadeIn();
              } else {
                _.each(data.driverLocations, showCab);
                $("#status_message").html(t("Nearest Cab Message", {
                  minutes: app.helpers.FormatSeconds(lowestETA, true)
                })).fadeIn();
              }
              if (Backbone.history.fragment === "!/request") {
                return _.delay(this.showCabs, this.pollInterval);
              }
            }
          }, this);
          return this.AjaxCall(ask, processData, attrs);
        case "StatusClient":
          processData = __bind(function(data, textStatus, jqXHR) {
            var bounds, cabLocation, locationSaved, point, userLocation;
            if (data.messageType === "OK") {
              switch (data.status) {
                case "completed":
                  this.removeCabs();
                  this.setStatus("rate");
                  return this.fetchTripDetails(data.tripID);
                case "open":
                  return this.setStatus("ready");
                case "begintrip":
                  this.setStatus("riding");
                  cabLocation = new google.maps.LatLng(data.latitude, data.longitude);
                  this.removeCabs();
                  this.pickup_icon.setMap(null);
                  point = new google.maps.Marker({
                    position: cabLocation,
                    map: this.map,
                    icon: this.cabMarker
                  });
                  this.cabs.push(point);
                  this.map.panTo(point.getPosition());
                  $("#rideName").html(data.driverName);
                  $("#ridePhone").html(data.driverMobile);
                  $("#ride_address_wrapper").hide();
                  if (Backbone.history.fragment === "!/request") {
                    return _.delay(this.AskDispatch, this.pollInterval, "StatusClient");
                  }
                  break;
                case "pending":
                  this.setStatus("searching");
                  if (Backbone.history.fragment === "!/request") {
                    return _.delay(this.AskDispatch, this.pollInterval, "StatusClient");
                  }
                  break;
                case "accepted":
                case "arrived":
                  if (data.status === "accepted") {
                    this.setStatus("waiting");
                    $("#status_message").html(t("Arrival ETA Message", {
                      minutes: app.helpers.FormatSeconds(data.eta, true)
                    }));
                  } else {
                    this.setStatus("arriving");
                    $("#status_message").html(t("Arriving Now Message"));
                  }
                  userLocation = new google.maps.LatLng(data.pickupLocation.latitude, data.pickupLocation.longitude);
                  cabLocation = new google.maps.LatLng(data.latitude, data.longitude);
                  this.pickup_icon.setPosition(userLocation);
                  this.removeCabs();
                  $("#rideName").html(data.driverName);
                  $("#ridePhone").html(data.driverMobile);
                  if ($("#rideAddress").html() === "") {
                    locationSaved = false;
                    _.each(USER.locations, __bind(function(location) {
                      if (parseFloat(location.latitude) === parseFloat(data.pickupLocation.latitude) && parseFloat(location.longitude) === parseFloat(data.pickupLocation.longitude)) {
                        return locationSaved = true;
                      }
                    }, this));
                    if (locationSaved) {
                      $("#addToFavButton").hide();
                    }
                    $("#pickupLat").val(data.pickupLocation.latitude);
                    $("#pickupLng").val(data.pickupLocation.longitude);
                    this.geocoder.geocode({
                      location: userLocation
                    }, __bind(function(result, status) {
                      $("#rideAddress").html(result[0].formatted_address);
                      return $("#favLocNickname").val("" + result[0].address_components[0].short_name + " " + result[0].address_components[1].short_name);
                    }, this));
                  }
                  point = new google.maps.Marker({
                    position: cabLocation,
                    map: this.map,
                    icon: this.cabMarker
                  });
                  this.cabs.push(point);
                  bounds = bounds = new google.maps.LatLngBounds();
                  bounds.extend(cabLocation);
                  bounds.extend(userLocation);
                  this.map.fitBounds(bounds);
                  if (Backbone.history.fragment === "!/request") {
                    return _.delay(this.AskDispatch, this.pollInterval, "StatusClient");
                  }
              }
            }
          }, this);
          return this.AjaxCall(ask, processData);
        case "Pickup":
          attrs = {
            latitude: this.pickup_icon.getPosition().lat(),
            longitude: this.pickup_icon.getPosition().lng()
          };
          processData = __bind(function(data, textStatus, jqXHR) {
            if (data.messageType === "Error") {
              return $("#status_message").html(data.description);
            } else {
              return this.AskDispatch("StatusClient");
            }
          }, this);
          return this.AjaxCall(ask, processData, attrs);
        case "PickupCanceledClient":
          processData = __bind(function(data, textStatus, jqXHR) {
            if (data.messageType === "OK") {
              return this.setStatus("ready");
            } else {
              return $("#status_message").html(data.description);
            }
          }, this);
          return this.AjaxCall(ask, processData, attrs);
        case "RatingDriver":
          attrs = {
            rating: options.rating
          };
          processData = __bind(function(data, textStatus, jqXHR) {
            if (data.messageType === "OK") {
              this.setStatus("init");
            } else {
              $("status_message").html(t("Rating Driver Failed"));
            }
            return this.HideSpinner();
          }, this);
          return this.AjaxCall(ask, processData, attrs);
        case "Feedback":
          attrs = {
            message: options.message
          };
          processData = __bind(function(data, textStatus, jqXHR) {
            if (data.messageType === "OK") {
              return alert("rated");
            }
          }, this);
          return this.AjaxCall(ask, processData, attrs);
      }
    };
    ClientsRequestView.prototype.AjaxCall = function(type, successCallback, attrs) {
      if (attrs == null) {
        attrs = {};
      }
      _.extend(attrs, {
        token: USER.token,
        messageType: type,
        app: "client",
        version: "1.0.60",
        device: "web"
      });
      return $.ajax({
        type: 'POST',
        url: DISPATCH + "/",
        processData: false,
        data: JSON.stringify(attrs),
        success: successCallback,
        dataType: 'json',
        error: __bind(function(jqXHR, textStatus, errorThrown) {
          $("#status_message").html(errorThrown);
          return this.HideSpinner();
        }, this)
      });
    };
    return ClientsRequestView;
  })();
}).call(this);
}, "views/clients/settings": function(exports, require, module) {(function() {
  var clientsSettingsTemplate;
  var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; }, __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  clientsSettingsTemplate = require('templates/clients/settings');
  exports.ClientsSettingsView = (function() {
    __extends(ClientsSettingsView, UberView);
    function ClientsSettingsView() {
      this.render = __bind(this.render, this);
      this.initialize = __bind(this.initialize, this);
      ClientsSettingsView.__super__.constructor.apply(this, arguments);
    }
    ClientsSettingsView.prototype.id = 'settings_view';
    ClientsSettingsView.prototype.className = 'view_container';
    ClientsSettingsView.prototype.events = {
      'submit #profile_pic_form': 'processPicUpload',
      'click #submit_pic': 'processPicUpload',
      'click a.setting_change': "changeTab",
      'submit #edit_info_form': "submitInfo",
      'click #change_password': 'changePass'
    };
    ClientsSettingsView.prototype.divs = {
      'info_div': "Information",
      'pic_div': "Picture"
    };
    ClientsSettingsView.prototype.pageTitle = t("Settings") + " | " + t("Uber");
    ClientsSettingsView.prototype.tabTitle = {
      'info_div': t("Information"),
      'pic_div': t("Picture")
    };
    ClientsSettingsView.prototype.initialize = function() {
      return this.mixin(require('web-lib/mixins/i18n_phone_form').i18nPhoneForm);
    };
    ClientsSettingsView.prototype.render = function(type) {
      if (type == null) {
        type = "info";
      }
      this.RefreshUserInfo(__bind(function() {
        var $el, alphabet;
        this.delegateEvents();
        this.HideSpinner();
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        $el = $(this.el);
        $(this.el).html(clientsSettingsTemplate({
          type: type
        }));
        $el.find("#" + type + "_div").show();
        $el.find("a[href='" + type + "_div']").parent().addClass("active");
        return document.title = "" + this.tabTitle[type + '_div'] + " " + this.pageTitle;
      }, this));
      this.delegateEvents();
      return this;
    };
    ClientsSettingsView.prototype.changeTab = function(e) {
      var $eTarget, $el, div, link, pageDiv, _i, _j, _len, _len2, _ref, _ref2;
      e.preventDefault();
      $eTarget = $(e.currentTarget);
      this.ClearGlobalStatus();
      $el = $(this.el);
      _ref = $el.find(".setting_change");
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        link = _ref[_i];
        $(link).parent().removeClass("active");
      }
      $eTarget.parent().addClass("active");
      _ref2 = _.keys(this.divs);
      for (_j = 0, _len2 = _ref2.length; _j < _len2; _j++) {
        div = _ref2[_j];
        $el.find("#" + div).hide();
      }
      pageDiv = $eTarget.attr('href');
      $el.find("#" + pageDiv).show();
      Backbone.history.navigate("!/settings/" + (this.divs[pageDiv].toLowerCase().replace(" ", "-")), false);
      document.title = "" + this.tabTitle[pageDiv] + " " + this.pageTitle;
      if (pageDiv === "loc_div") {
        try {
          google.maps.event.trigger(this.map, 'resize');
          return this.map.fitBounds(this.bounds);
        } catch (_e) {}
      }
    };
    ClientsSettingsView.prototype.submitInfo = function(e) {
      var $e, attrs, client, options;
      $('#global_status').find('.success_message').text('');
      $('#global_status').find('.error_message').text('');
      $('.error_message').text('');
      e.preventDefault();
      $e = $(e.currentTarget);
      attrs = $e.serializeToJson();
      attrs['mobile_country_id'] = this.$('#mobile_country_id').val();
      if (attrs['password'] === '') {
        delete attrs['password'];
      }
      options = {
        success: __bind(function(response) {
          this.ShowSuccess(t("Information Update Succeeded"));
          return this.RefreshUserInfo();
        }, this),
        error: __bind(function(model, data) {
          var errors;
          if (data.status === 406) {
            errors = JSON.parse(data.responseText);
            return _.each(_.keys(errors), function(field) {
              return $("#" + field).parent().find('span.error_message').text(errors[field]);
            });
          } else {
            return this.ShowError(t("Information Update Failed"));
          }
        }, this),
        type: "PUT"
      };
      client = new app.models.client({
        id: USER.id
      });
      return client.save(attrs, options);
    };
    ClientsSettingsView.prototype.changePass = function(e) {
      e.preventDefault();
      $(e.currentTarget).hide();
      return $("#password").show();
    };
    ClientsSettingsView.prototype.processPicUpload = function(e) {
      e.preventDefault();
      this.ShowSpinner("submit");
      return $.ajaxFileUpload({
        url: API + '/user_pictures',
        secureuri: false,
        fileElementId: 'picture',
        data: {
          token: USER.token
        },
        dataType: 'json',
        complete: __bind(function(data, status) {
          this.HideSpinner();
          if (status === 'success') {
            this.ShowSuccess(t("Picture Update Succeeded"));
            return this.RefreshUserInfo(__bind(function() {
              return $("#settingsProfPic").attr("src", USER.picture_url + ("?" + (Math.floor(Math.random() * 1000))));
            }, this));
          } else {
            if (data.error) {
              return this.ShowError(data.error);
            } else {
              return this.ShowError("Picture Update Failed");
            }
          }
        }, this)
      });
    };
    return ClientsSettingsView;
  })();
}).call(this);
}, "views/clients/sign_up": function(exports, require, module) {(function() {
  var clientsSignUpTemplate;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  }, __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };
  clientsSignUpTemplate = require('templates/clients/sign_up');
  exports.ClientsSignUpView = (function() {
    __extends(ClientsSignUpView, UberView);
    function ClientsSignUpView() {
      ClientsSignUpView.__super__.constructor.apply(this, arguments);
    }
    ClientsSignUpView.prototype.id = 'signup_view';
    ClientsSignUpView.prototype.className = 'view_container';
    ClientsSignUpView.prototype.initialize = function() {
      this.mixin(require('web-lib/mixins/i18n_phone_form').i18nPhoneForm);
      return $('#location_country').live('change', function() {
        if (!$('#mobile').val()) {
          return $('#mobile_country').find("option[value=" + ($(this).val()) + "]").attr('selected', 'selected').end().trigger('change');
        }
      });
    };
    ClientsSignUpView.prototype.events = {
      'submit form': 'signup',
      'click button': 'signup',
      'change #card_number': 'showCardType',
      'change #location_country': 'countryChange'
    };
    ClientsSignUpView.prototype.render = function(invite) {
      this.HideSpinner();
      $(this.el).html(clientsSignUpTemplate({
        invite: invite
      }));
      return this;
    };
    ClientsSignUpView.prototype.signup = function(e) {
      var $el, attrs, client, error_messages, options;
      e.preventDefault();
      $el = $("form");
      $el.find('#terms_error').hide();
      if (!$el.find('#signup_terms input[type=checkbox]').attr('checked')) {
        $('#spinner.submit').hide();
        $el.find('#terms_error').show();
        return;
      }
      error_messages = $el.find('.error_message').html("");
      attrs = {
        first_name: $el.find('#first_name').val(),
        last_name: $el.find('#last_name').val(),
        email: $el.find('#email').val(),
        password: $el.find('#password').val(),
        location_country: $el.find('#location_country option:selected').attr('data-iso2'),
        location: $el.find('#location').val(),
        language: $el.find('#language').val(),
        mobile_country: $el.find('#mobile_country option:selected').attr('data-iso2'),
        mobile: $el.find('#mobile').val(),
        card_number: $el.find('#card_number').val(),
        card_expiration_month: $el.find('#card_expiration_month').val(),
        card_expiration_year: $el.find('#card_expiration_year').val(),
        card_code: $el.find('#card_code').val(),
        use_case: $el.find('#use_case').val(),
        promotion_code: $el.find('#promotion_code').val()
      };
      options = {
        statusCode: {
          200: function(response) {
            $.cookie('token', response.token);
            amplify.store('USERjson', response);
            app.refreshMenu();
            return app.routers.clients.navigate('!/dashboard', true);
          },
          406: function(e) {
            var error, errors, _i, _len, _ref, _results;
            errors = JSON.parse(e.responseText);
            _ref = _.keys(errors);
            _results = [];
            for (_i = 0, _len = _ref.length; _i < _len; _i++) {
              error = _ref[_i];
              _results.push($('#' + error).parent().find('span').html($('#' + error).parent().find('span').html() + " " + errors[error]));
            }
            return _results;
          }
        },
        complete: __bind(function(response) {
          return this.HideSpinner();
        }, this)
      };
      client = new app.models.client;
      $('.spinner#submit').show();
      return client.save(attrs, options);
    };
    ClientsSignUpView.prototype.countryChange = function(e) {
      var $e;
      $e = $(e.currentTarget);
      return $("#mobile_country").val($e.val()).trigger('change');
    };
    ClientsSignUpView.prototype.showCardType = function(e) {
      var $el, reAmerica, reDiscover, reMaster, reVisa, validCard;
      reVisa = /^4\d{3}-?\d{4}-?\d{4}-?\d{4}$/;
      reMaster = /^5[1-5]\d{2}-?\d{4}-?\d{4}-?\d{4}$/;
      reAmerica = /^6011-?\d{4}-?\d{4}-?\d{4}$/;
      reDiscover = /^3[4,7]\d{13}$/;
      $el = $("#card_logos_signup");
      validCard = false;
      if (e.currentTarget.value.match(reVisa)) {
        $el.find("#overlay_left").css('width', "0px");
        return $el.find("#overlay_right").css('width', "75%");
      } else if (e.currentTarget.value.match(reMaster)) {
        $el.find("#overlay_left").css('width', "25%");
        return $el.find("#overlay_right").css('width', "50%");
      } else if (e.currentTarget.value.match(reAmerica)) {
        $el.find("#overlay_left").css('width', "75%");
        $el.find("#overlay_right").css('width', "0px");
        return console.log("amex");
      } else if (e.currentTarget.value.match(reDiscover)) {
        $el.find("#overlay_left").css('width', "50%");
        return $el.find("#overlay_right").css('width', "25%");
      } else {
        $el.find("#overlay_left").css('width', "0px");
        return $el.find("#overlay_right").css('width', "0px");
      }
    };
    return ClientsSignUpView;
  })();
}).call(this);
}, "views/clients/trip_detail": function(exports, require, module) {(function() {
  var clientsTripDetailTemplate;
  var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; }, __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  clientsTripDetailTemplate = require('templates/clients/trip_detail');
  exports.TripDetailView = (function() {
    __extends(TripDetailView, UberView);
    function TripDetailView() {
      this.resendReceipt = __bind(this.resendReceipt, this);
      TripDetailView.__super__.constructor.apply(this, arguments);
    }
    TripDetailView.prototype.id = 'trip_detail_view';
    TripDetailView.prototype.className = 'view_container';
    TripDetailView.prototype.events = {
      'click a#fare_review': 'showFareReview',
      'click #fare_review_hide': 'hideFareReview',
      'submit #form_review_form': 'submitFareReview',
      'click #submit_fare_review': 'submitFareReview',
      'click .resendReceipt': 'resendReceipt'
    };
    TripDetailView.prototype.render = function(id) {
      if (id == null) {
        id = 'invalid';
      }
      this.ReadUserInfo();
      this.HideSpinner();
      this.model = new app.models.trip({
        id: id
      });
      this.model.fetch({
        data: {
          relationships: 'points,driver,city.country'
        },
        dataType: 'json',
        success: __bind(function() {
          var trip;
          trip = this.model;
          $(this.el).html(clientsTripDetailTemplate({
            trip: trip
          }));
          this.RequireMaps(__bind(function() {
            var bounds, endPos, map, myOptions, path, polyline, startPos;
            bounds = new google.maps.LatLngBounds();
            path = [];
            _.each(this.model.get('points'), __bind(function(point) {
              path.push(new google.maps.LatLng(point.lat, point.lng));
              return bounds.extend(_.last(path));
            }, this));
            myOptions = {
              zoom: 12,
              center: path[0],
              mapTypeId: google.maps.MapTypeId.ROADMAP,
              zoomControl: false,
              rotateControl: false,
              panControl: false,
              mapTypeControl: false,
              scrollwheel: false
            };
            map = new google.maps.Map(document.getElementById("trip_details_map"), myOptions);
            map.fitBounds(bounds);
            startPos = new google.maps.Marker({
              position: _.first(path),
              map: map,
              title: t("Trip started here"),
              icon: 'https://uber-static.s3.amazonaws.com/marker_start.png'
            });
            endPos = new google.maps.Marker({
              position: _.last(path),
              map: map,
              title: t("Trip ended here"),
              icon: 'https://uber-static.s3.amazonaws.com/marker_end.png'
            });
            startPos.setMap(map);
            endPos.setMap(map);
            polyline = new google.maps.Polyline({
              path: path,
              strokeColor: '#003F87',
              strokeOpacity: 1,
              strokeWeight: 5
            });
            return polyline.setMap(map);
          }, this));
          return this.HideSpinner();
        }, this)
      });
      this.ShowSpinner('load');
      this.delegateEvents();
      return this;
    };
    TripDetailView.prototype.showFareReview = function(e) {
      e.preventDefault();
      $('#fare_review_box').slideDown();
      return $('#fare_review').hide();
    };
    TripDetailView.prototype.hideFareReview = function(e) {
      e.preventDefault();
      $('#fare_review_box').slideUp();
      return $('#fare_review').show();
    };
    TripDetailView.prototype.submitFareReview = function(e) {
      var attrs, errorMessage, id, options;
      e.preventDefault();
      errorMessage = $(".error_message");
      errorMessage.hide();
      id = $("#tripid").val();
      this.model = new app.models.trip({
        id: id
      });
      attrs = {
        note: $('#form_review_message').val(),
        note_type: 'client_fare_review'
      };
      options = {
        success: __bind(function(response) {
          $(".success_message").fadeIn();
          return $("#fare_review_form_wrapper").slideUp();
        }, this),
        error: __bind(function(error) {
          return errorMessage.fadeIn();
        }, this)
      };
      return this.model.save(attrs, options);
    };
    TripDetailView.prototype.resendReceipt = function(e) {
      var $e;
      e.preventDefault();
      $e = $(e.currentTarget);
      this.$(".resendReceiptSuccess").empty().show();
      this.$(".resentReceiptError").empty().show();
      e.preventDefault();
      $('#spinner').show();
      return $.ajax('/api/trips/func/resend_receipt', {
        data: {
          token: $.cookie('token'),
          trip_id: this.model.id
        },
        type: 'POST',
        complete: __bind(function(xhr) {
          var response;
          response = JSON.parse(xhr.responseText);
          $('#spinner').hide();
          switch (xhr.status) {
            case 200:
              this.$(".resendReceiptSuccess").html("Receipt has been emailed");
              return this.$(".resendReceiptSuccess").fadeOut(2000);
            default:
              this.$(".resendReceiptError").html("Receipt has failed to be emailed");
              return this.$(".resendReceiptError").fadeOut(2000);
          }
        }, this)
      });
    };
    return TripDetailView;
  })();
}).call(this);
}, "views/shared/menu": function(exports, require, module) {(function() {
  var menuTemplate;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  menuTemplate = require('templates/shared/menu');
  exports.SharedMenuView = (function() {
    __extends(SharedMenuView, Backbone.View);
    function SharedMenuView() {
      SharedMenuView.__super__.constructor.apply(this, arguments);
    }
    SharedMenuView.prototype.id = 'menu_view';
    SharedMenuView.prototype.render = function() {
      var type;
      if ($.cookie('token') === null) {
        type = 'guest';
      } else {
        type = 'client';
      }
      $(this.el).html(menuTemplate({
        type: type
      }));
      return this;
    };
    return SharedMenuView;
  })();
}).call(this);
}, "web-lib/collections/countries": function(exports, require, module) {(function() {
  var UberCollection;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  UberCollection = require('web-lib/uber_collection').UberCollection;
  exports.CountriesCollection = (function() {
    __extends(CountriesCollection, UberCollection);
    function CountriesCollection() {
      CountriesCollection.__super__.constructor.apply(this, arguments);
    }
    CountriesCollection.prototype.model = app.models.country;
    CountriesCollection.prototype.url = '/countries';
    return CountriesCollection;
  })();
}).call(this);
}, "web-lib/collections/vehicle_types": function(exports, require, module) {(function() {
  var UberCollection, vehicleType, _ref;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  UberCollection = require('web-lib/uber_collection').UberCollection;
  vehicleType = (typeof app !== "undefined" && app !== null ? (_ref = app.models) != null ? _ref.vehicleType : void 0 : void 0) || require('models/vehicle_type').VehicleType;
  exports.VehicleTypesCollection = (function() {
    __extends(VehicleTypesCollection, UberCollection);
    function VehicleTypesCollection() {
      VehicleTypesCollection.__super__.constructor.apply(this, arguments);
    }
    VehicleTypesCollection.prototype.model = vehicleType;
    VehicleTypesCollection.prototype.url = '/vehicle_types';
    VehicleTypesCollection.prototype.defaultColumns = ['id', 'created_at', 'updated_at', 'deleted_at', 'created_by_user_id', 'updated_by_user_id', 'city_id', 'type', 'make', 'model', 'capacity', 'minimum_year', 'actions'];
    VehicleTypesCollection.prototype.tableColumns = function(cols) {
      var actions, c, capacity, city_id, columnValues, created_at, created_by_user_id, deleted_at, headerRow, id, make, minimum_year, model, type, updated_at, updated_by_user_id, _i, _len;
      id = {
        sTitle: 'Id'
      };
      created_at = {
        sTitle: 'Created At (UTC)',
        'sType': 'string'
      };
      updated_at = {
        sTitle: 'Updated At (UTC)',
        'sType': 'string'
      };
      deleted_at = {
        sTitle: 'Deleted At (UTC)',
        'sType': 'string'
      };
      created_by_user_id = {
        sTitle: 'Created By'
      };
      updated_by_user_id = {
        sTitle: 'Updated By'
      };
      city_id = {
        sTitle: 'City'
      };
      type = {
        sTitle: 'Type'
      };
      make = {
        sTitle: 'Make'
      };
      model = {
        sTitle: 'Model'
      };
      capacity = {
        sTitle: 'Capacity'
      };
      minimum_year = {
        sTitle: 'Min. Year'
      };
      actions = {
        sTitle: 'Actions'
      };
      columnValues = {
        id: id,
        created_at: created_at,
        updated_at: updated_at,
        deleted_at: deleted_at,
        created_by_user_id: created_by_user_id,
        updated_by_user_id: updated_by_user_id,
        city_id: city_id,
        type: type,
        make: make,
        model: model,
        capacity: capacity,
        minimum_year: minimum_year,
        actions: actions
      };
      headerRow = [];
      for (_i = 0, _len = cols.length; _i < _len; _i++) {
        c = cols[_i];
        if (columnValues[c]) {
          headerRow.push(columnValues[c]);
        }
      }
      return headerRow;
    };
    return VehicleTypesCollection;
  })();
}).call(this);
}, "web-lib/helpers": function(exports, require, module) {(function() {
  var __indexOf = Array.prototype.indexOf || function(item) {
    for (var i = 0, l = this.length; i < l; i++) {
      if (this[i] === item) return i;
    }
    return -1;
  }, __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };
  exports.helpers = {
    pin: function(num, color) {
      if (color == null) {
        color = 'FF0000';
      }
      return "<img src=\"http://chart.apis.google.com/chart?chst=d_map_pin_letter&chld=" + num + "|" + color + "|000000\" width=\"14\" height=\"22\" />";
    },
    reverseGeocode: function(latitude, longitude) {
      if (latitude && longitude) {
        return "<span data-point=" + (JSON.stringify({
          latitude: latitude,
          longitude: longitude
        })) + ">" + latitude + ", " + longitude + "</span>";
      } else {
        return '';
      }
    },
    linkedName: function(model) {
      var first_name, id, last_name, role, url;
      role = model.role || model.get('role');
      id = model.id || model.get('id');
      first_name = model.first_name || model.get('first_name');
      last_name = model.last_name || model.get('last_name');
      url = "/" + role + "s/" + id;
      return "<a href=\"#" + url + "\">" + first_name + " " + last_name + "</a>";
    },
    linkedVehicle: function(vehicle, vehicleType) {
      return "<a href=\"/#/vehicles/" + vehicle.id + "\">      " + (vehicleType != null ? vehicleType.get('make') : void 0) + "      " + (vehicleType != null ? vehicleType.get('model') : void 0) + "      " + (vehicle.get('year')) + "    </a>";
    },
    linkedUserId: function(userType, userId) {
      return "<a href=\"#!/" + userType + "/" + userId + "\" data-user-type=\"" + userType + "\" data-user-id=\"" + userId + "\">" + userType + " " + userId + "</a>";
    },
    timeDelta: function(start, end) {
      var delta;
      if (typeof start === 'string') {
        start = this.parseDate(start);
      }
      if (typeof end === 'string') {
        end = this.parseDate(end);
      }
      if (end && start) {
        delta = end.getTime() - start.getTime();
        return this.formatSeconds(delta / 1000);
      } else {
        return '00:00';
      }
    },
    formatSeconds: function(s) {
      var minutes, seconds;
      s = Math.floor(s);
      minutes = Math.floor(s / 60);
      seconds = s - minutes * 60;
      return "" + (this.leadingZero(minutes)) + ":" + (this.leadingZero(seconds));
    },
    formatCurrency: function(strValue, reverseSign, currency) {
      var currency_locale, lc, mf;
      if (reverseSign == null) {
        reverseSign = false;
      }
      if (currency == null) {
        currency = null;
      }
      strValue = String(strValue);
      if (reverseSign) {
        strValue = ~strValue.indexOf('-') ? strValue.split('-').join('') : ['-', strValue].join('');
      }
      currency_locale = i18n.currencyToLocale[currency];
      try {
        if (!(currency_locale != null) || currency_locale === i18n.locale) {
          return i18n.jsworld.mf.format(strValue);
        } else {
          lc = new jsworld.Locale(POSIX_LC[currency_locale]);
          mf = new jsworld.MonetaryFormatter(lc);
          return mf.format(strValue);
        }
      } catch (error) {
        i18n.log(error);
        return strValue;
      }
    },
    formatTripFare: function(trip, type) {
      var _ref, _ref2;
      if (type == null) {
        type = "fare";
      }
      if (!trip.get('fare')) {
        return 'n/a';
      }
      if (((_ref = trip.get('fare_breakdown_local')) != null ? _ref.currency : void 0) != null) {
        return app.helpers.formatCurrency(trip.get("" + type + "_local"), false, (_ref2 = trip.get('fare_breakdown_local')) != null ? _ref2.currency : void 0);
      } else if (trip.get("" + type + "_string") != null) {
        return trip.get("" + type + "_string");
      } else if (trip.get("" + type + "_local") != null) {
        return trip.get("" + type + "_local");
      } else {
        return 'n/a';
      }
    },
    formatPhoneNumber: function(phoneNumber, countryCode) {
      if (countryCode == null) {
        countryCode = "+1";
      }
      if (phoneNumber != null) {
        phoneNumber = String(phoneNumber);
        switch (countryCode) {
          case '+1':
            return countryCode + ' ' + phoneNumber.substring(0, 3) + '-' + phoneNumber.substring(3, 6) + '-' + phoneNumber.substring(6, 10);
          case '+33':
            return countryCode + ' ' + phoneNumber.substring(0, 1) + ' ' + phoneNumber.substring(1, 3) + ' ' + phoneNumber.substring(3, 5) + ' ' + phoneNumber.substring(5, 7) + ' ' + phoneNumber.substring(7, 9);
          default:
            countryCode + phoneNumber;
        }
      }
      return "" + countryCode + " " + phoneNumber;
    },
    parseDate: function(d, cityTime, tz) {
      var city_filter, parsed, _ref;
      if (cityTime == null) {
        cityTime = true;
      }
      if (tz == null) {
        tz = null;
      }
      if (((_ref = !d.substr(-6, 1)) === '+' || _ref === '-') || d.length === 19) {
        d += '+00:00';
      }
      if (/(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.test(d)) {
        parsed = d.match(/(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/);
        d = new Date();
        d.setUTCFullYear(parsed[1]);
        d.setUTCMonth(parsed[2] - 1);
        d.setUTCDate(parsed[3]);
        d.setUTCHours(parsed[4]);
        d.setUTCMinutes(parsed[5]);
        d.setUTCSeconds(parsed[6]);
      } else {
        d = Date.parse(d);
      }
      if (typeof d === 'number') {
        d = new Date(d);
      }
      d = new timezoneJS.Date(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds(), 'Etc/UTC');
      if (tz) {
        d.convertToTimezone(tz);
      } else if (cityTime) {
        city_filter = $.cookie('city_filter');
        if (city_filter) {
          tz = $("#city_filter option[value=" + city_filter + "]").attr('data-timezone');
          if (tz) {
            d.convertToTimezone(tz);
          }
        }
      }
      return d;
    },
    dateToTimezone: function(d) {
      var city_filter, tz;
      d = new timezoneJS.Date(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds(), 'Etc/UTC');
      city_filter = $.cookie('city_filter');
      if (city_filter) {
        tz = $("#city_filter option[value=" + city_filter + "]").attr('data-timezone');
        d.convertToTimezone(tz);
      }
      return d;
    },
    fixAMPM: function(d, formatted) {
      if (d.hours >= 12) {
        return formatted.replace(/\b[AP]M\b/, 'PM');
      } else {
        return formatted.replace(/\b[AP]M\b/, 'AM');
      }
    },
    formatDate: function(d, time, timezone) {
      var formatted;
      if (time == null) {
        time = true;
      }
      if (timezone == null) {
        timezone = null;
      }
      d = this.parseDate(d, true, timezone);
      formatted = time ? ("" + (i18n.jsworld.dtf.formatDate(d)) + " ") + this.formatTime(d, d.getTimezoneInfo()) : i18n.jsworld.dtf.formatDate(d);
      return this.fixAMPM(d, formatted);
    },
    formatDateLong: function(d, time, timezone) {
      if (time == null) {
        time = true;
      }
      if (timezone == null) {
        timezone = null;
      }
      d = this.parseDate(d, true, timezone);
      timezone = d.getTimezoneInfo().tzAbbr;
      if (time) {
        return (i18n.jsworld.dtf.formatDateTime(d)) + (" " + timezone);
      } else {
        return i18n.jsworld.dtf.formatDate(d);
      }
    },
    formatTimezoneJSDate: function(d) {
      var day, hours, jsDate, minutes, month, year;
      year = d.getFullYear();
      month = this.leadingZero(d.getMonth());
      day = this.leadingZero(d.getDate());
      hours = this.leadingZero(d.getHours());
      minutes = this.leadingZero(d.getMinutes());
      jsDate = new Date(year, month, day, hours, minutes, 0);
      return jsDate.toDateString();
    },
    formatTime: function(d, timezone) {
      var formatted;
      if (timezone == null) {
        timezone = null;
      }
      formatted = ("" + (i18n.jsworld.dtf.formatTime(d))) + (timezone != null ? " " + (timezone != null ? timezone.tzAbbr : void 0) : "");
      return this.fixAMPM(d, formatted);
    },
    formatISODate: function(d) {
      var pad;
      pad = function(n) {
        if (n < 10) {
          return '0' + n;
        }
        return n;
      };
      return d.getUTCFullYear() + '-' + pad(d.getUTCMonth() + 1) + '-' + pad(d.getUTCDate()) + 'T' + pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes()) + ':' + pad(d.getUTCSeconds()) + 'Z';
    },
    formatExpDate: function(d) {
      var month, year;
      d = this.parseDate(d);
      year = d.getFullYear();
      month = this.leadingZero(d.getMonth() + 1);
      return "" + year + "-" + month;
    },
    formatLatLng: function(lat, lng, precision) {
      if (precision == null) {
        precision = 8;
      }
      return parseFloat(lat).toFixed(precision) + ',' + parseFloat(lng).toFixed(precision);
    },
    leadingZero: function(num) {
      if (num < 10) {
        return "0" + num;
      } else {
        return num;
      }
    },
    roundNumber: function(num, dec) {
      return Math.round(num * Math.pow(10, dec)) / Math.pow(10, dec);
    },
    notesToHTML: function(notes) {
      var i, note, notesHTML, _i, _len;
      notesHTML = '';
      i = 1;
      if (notes) {
        for (_i = 0, _len = notes.length; _i < _len; _i++) {
          note = notes[_i];
          notesHTML += "<strong>" + note['userid'] + "</strong> &nbsp;&nbsp;&nbsp; " + (this.formatDate(note['created_at'])) + "<p>" + note['note'] + "</p>";
          notesHTML += "<br>";
        }
      }
      return notesHTML.replace("'", '&quote');
    },
    formatPhone: function(n) {
      var parts, phone, regexObj;
      n = "" + n;
      regexObj = /^(?:\+?1[-. ]?)?(?:\(?([0-9]{3})\)?[-. ]?)?([0-9]{3})[-. ]?([0-9]{4})$/;
      if (regexObj.test(n)) {
        parts = n.match(regexObj);
        phone = "";
        if (parts[1]) {
          phone += "(" + parts[1] + ") ";
        }
        phone += "" + parts[2] + "-" + parts[3];
      } else {
        phone = n;
      }
      return phone;
    },
    usStates: ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'],
    onboardingPages: ['applied', 'ready_to_interview', 'pending_interview', 'interviewed', 'accepted', 'ready_to_onboard', 'pending_onboarding', 'active', 'waitlisted', 'rejected'],
    driverBreadCrumb: function(loc, model) {
      var onboardingPage, out, _i, _len, _ref;
      out = "<a href='#/driver_ops/summary'>Drivers</a> > ";
      if (!(model != null)) {
        out += "<select name='onboardingPage' id='onboardingPageSelector'>";
        _ref = this.onboardingPages;
        for (_i = 0, _len = _ref.length; _i < _len; _i++) {
          onboardingPage = _ref[_i];
          out += "<option value='" + onboardingPage + "' " + (onboardingPage === loc ? "selected" : void 0) + ">" + (this.onboardingUrlToName(onboardingPage)) + "</option>";
        }
        out += "</select>";
      } else {
        out += "<a href='#/driver_ops/" + (model.get('driver_status')) + "'>" + (this.onboardingUrlToName(model.get('driver_status'))) + "</a>";
        out += " > " + (this.linkedName(model)) + " (" + (model.get('role')) + ") #" + (model.get('id'));
      }
      return out;
    },
    onboardingUrlToName: function(url) {
      return url != null ? url.replace(/_/g, " ").replace(/(^|\s)([a-z])/g, function(m, p1, p2) {
        return p1 + p2.toUpperCase();
      }) : void 0;
    },
    formatVehicle: function(vehicle) {
      if (vehicle.get('make') && vehicle.get('model') && vehicle.get('license_plate')) {
        return "" + (vehicle.get('make')) + " " + (vehicle.get('model')) + " (" + (vehicle.get('license_plate')) + ")";
      }
    },
    docArbitraryFields: function(docName, cityDocs) {
      var doc, field, out, _i, _j, _len, _len2, _ref;
      out = "";
      for (_i = 0, _len = cityDocs.length; _i < _len; _i++) {
        doc = cityDocs[_i];
        if (doc.name === docName && __indexOf.call(_.keys(doc), "metaFields") >= 0) {
          _ref = doc.metaFields;
          for (_j = 0, _len2 = _ref.length; _j < _len2; _j++) {
            field = _ref[_j];
            out += "" + field.label + ": <input type='text' name='" + field.name + "' class='arbitraryField'><br>";
          }
        }
      }
      return out;
    },
    capitaliseFirstLetter: function(string) {
      return string.charAt(0).toUpperCase() + string.slice(1);
    },
    createDocUploadForm: function(docName, driverId, vehicleId, cityMeta, vehicleName, expirationRequired) {
      var ddocs, expDropdowns, pdocs, vdocs;
      if (driverId == null) {
        driverId = "None";
      }
      if (vehicleId == null) {
        vehicleId = "None";
      }
      if (cityMeta == null) {
        cityMeta = [];
      }
      if (vehicleName == null) {
        vehicleName = false;
      }
      if (expirationRequired == null) {
        expirationRequired = false;
      }
      ddocs = cityMeta["driverRequiredDocs"] || [];
      pdocs = cityMeta["partnerRequiredDocs"] || [];
      vdocs = cityMeta["vehicleRequiredDocs"] || [];
      expDropdowns = "Expiration Date:\n<select name=\"expiration-year\">\n  <option value=\"2011\">2011</option>\n  <option value=\"2012\">2012</option>\n  <option value=\"2013\">2013</option>\n  <option value=\"2014\">2014</option>\n  <option value=\"2015\">2015</option>\n  <option value=\"2016\">2016</option>\n  <option value=\"2017\">2017</option>\n  <option value=\"2018\">2018</option>\n</select> -\n<select name=\"expiration-month\">\n  <option value=\"01\">01</option>\n  <option value=\"02\">02</option>\n  <option value=\"03\">03</option>\n  <option value=\"04\">04</option>\n  <option value=\"05\">05</option>\n  <option value=\"06\">06</option>\n  <option value=\"07\">07</option>\n  <option value=\"08\">08</option>\n  <option value=\"09\">09</option>\n  <option value=\"10\">10</option>\n  <option value=\"11\">11</option>\n  <option value=\"12\">12</option>\n</select>";
      return "  <form class=\"documentuploadform\">\n  <div>\n    <input type=\"hidden\" name=\"fileName\" value=\"" + docName + "\">\n    <input type=\"hidden\" name=\"driver_id\" value=\"" + driverId + "\">\n    <input type=\"hidden\" name=\"vehicle_id\" value=\"" + vehicleId + "\">\n\n    <div>\n      <strong>" + (vehicleName ? vehicleName : "") + " " + docName + "</strong>\n    </div>\n\n    <div>\n      <input type=\"file\" name=\"uploadContent\" id=\"" + (vehicleId !== "None" ? "vehicle_" + vehicleId + "_" : "") + "doc_upload_" + (docName.replace(/[\W]/g, "_")) + "\">\n    </div>\n\n    <div class=\"expiration\">\n      " + (expirationRequired ? expDropdowns : "") + "\n    </div>\n\n    <div>\n      " + (app.helpers.docArbitraryFields(docName, _.union(ddocs, pdocs, vdocs))) + "\n    </div>\n\n    <div>\n      <input type=\"submit\" value=\"Upload\">\n    </div>\n\n  </div>\n</form>";
    },
    countrySelector: function(name, options) {
      var countries, countryCodePrefix, defaultOptions;
      if (options == null) {
        options = {};
      }
      defaultOptions = {
        selectedKey: 'telephone_code',
        selectedValue: '+1',
        silent: false
      };
      _.extend(defaultOptions, options);
      options = defaultOptions;
      countries = new app.collections.countries();
      countries.fetch({
        data: {
          limit: 300
        },
        success: function(countries) {
          var $option, $select, country, selected, _i, _len, _ref;
          selected = false;
          _ref = countries.models || [];
          for (_i = 0, _len = _ref.length; _i < _len; _i++) {
            country = _ref[_i];
            $select = $("select[name=" + name + "]");
            $option = $('<option></option>').val(country.id).attr('data-iso2', country.get('iso2')).attr('data-prefix', country.get('telephone_code')).html(country.get('name'));
            if (country.get(options.selectedKey) === options.selectedValue && !selected) {
              selected = true;
              $option.attr('selected', 'selected');
            }
            $select.append($option);
          }
          if (selected && !options.silent) {
            return $select.val(options.selected).trigger('change');
          }
        }
      });
      countryCodePrefix = options.countryCodePrefix ? "data-country-code-prefix='" + options.countryCodePrefix + "'" : '';
      return "<select name='" + name + "' id='" + name + "' " + countryCodePrefix + " " + (options.disabled ? 'disabled="disabled"' : "") + "></select>";
    },
    missingDocsOnDriver: function(driver) {
      var city, docsReq, documents, partnerDocs;
      city = driver.get('city');
      documents = driver.get('documents');
      if ((city != null) && (documents != null)) {
        docsReq = _.pluck(city != null ? city.get('meta')["driverRequiredDocs"] : void 0, "name");
        if (driver.get('role') === "partner") {
          partnerDocs = _.pluck(city != null ? city.get('meta')["partnerRequiredDocs"] : void 0, "name");
          docsReq = _.union(docsReq, partnerDocs);
        }
        return _.reject(docsReq, __bind(function(doc) {
          return __indexOf.call((documents != null ? documents.pluck("name") : void 0) || [], doc) >= 0;
        }, this));
      } else {
        return [];
      }
    }
  };
}).call(this);
}, "web-lib/i18n": function(exports, require, module) {(function() {
  exports.i18n = {
    defaultLocale: 'en_US',
    cookieName: '_LOCALE_',
    locales: {
      'en_US': "English (US)",
      'fr_FR': "Français"
    },
    currencyToLocale: {
      'USD': 'en_US',
      'EUR': 'fr_FR'
    },
    logglyKey: 'd2d5a9bc-7ebe-4538-a180-81e62c705b1b',
    logglyHost: 'https://logs.loggly.com',
    init: function() {
      this.castor = new window.loggly({
        url: this.logglyHost + '/inputs/' + this.logglyKey + '?rt=1',
        level: 'error'
      });
      this.setLocale($.cookie(this.cookieName) || this.defaultLocale);
      window.t = _.bind(this.t, this);
      this.loadLocaleTranslations(this.locale);
      if (!(this[this.defaultLocale] != null)) {
        return this.loadLocaleTranslations(this.defaultLocale);
      }
    },
    loadLocaleTranslations: function(locale) {
      var loadPaths, path, _i, _len, _results;
      loadPaths = ['web-lib/translations/' + locale, 'web-lib/translations/' + locale.slice(0, 2), 'translations/' + locale, 'translations/' + locale.slice(0, 2)];
      _results = [];
      for (_i = 0, _len = loadPaths.length; _i < _len; _i++) {
        path = loadPaths[_i];
        locale = path.substring(path.lastIndexOf('/') + 1);
        if (this[locale] == null) {
          this[locale] = {};
        }
        _results.push((function() {
          try {
            return _.extend(this[locale], require(path).translations);
          } catch (error) {

          }
        }).call(this));
      }
      return _results;
    },
    getLocale: function() {
      return this.locale;
    },
    setLocale: function(locale) {
      var message, parts, _ref;
      parts = locale.split('_');
      this.locale = parts[0].toLowerCase();
      if (parts.length > 1) {
        this.locale += "_" + (parts[1].toUpperCase());
      }
      if (this.locale) {
        $.cookie(this.cookieName, this.locale, {
          path: '/',
          domain: '.uber.com'
        });
      }
      try {
        ((_ref = this.jsworld) != null ? _ref : this.jsworld = {}).lc = new jsworld.Locale(POSIX_LC[this.locale]);
        this.jsworld.mf = new jsworld.MonetaryFormatter(this.jsworld.lc);
        this.jsworld.nf = new jsworld.NumericFormatter(this.jsworld.lc);
        this.jsworld.dtf = new jsworld.DateTimeFormatter(this.jsworld.lc);
        this.jsworld.np = new jsworld.NumericParser(this.jsworld.lc);
        this.jsworld.mp = new jsworld.MonetaryParser(this.jsworld.lc);
        return this.jsworld.dtp = new jsworld.DateTimeParser(this.jsworld.lc);
      } catch (error) {
        message = 'JsWorld error with locale: ' + this.locale;
        return this.log({
          message: message,
          error: error
        });
      }
    },
    getTemplate: function(id) {
      var _ref, _ref2;
      return ((_ref = this[this.locale]) != null ? _ref[id] : void 0) || ((_ref2 = this[this.locale.slice(0, 2)]) != null ? _ref2[id] : void 0);
    },
    getTemplateDefault: function(id) {
      var _ref, _ref2;
      return ((_ref = this[this.defaultLocale]) != null ? _ref[id] : void 0) || ((_ref2 = this[this.defaultLocale.slice(0, 2)]) != null ? _ref2[id] : void 0);
    },
    getTemplateOrDefault: function(id) {
      return this.getTemplate(id) || this.getTemplateDefault(id);
    },
    t: function(id, vars) {
      var errStr, locale, template;
      if (vars == null) {
        vars = {};
      }
      locale = this.getLocale();
      template = this.getTemplate(id);
      if (template == null) {
        if (/dev|test/.test(window.location.host)) {
          template = "(?) " + id;
        } else {
          template = this.getTemplateDefault(id);
        }
        errStr = "Missing [" + locale + "] translation for [" + id + "] at [" + window.location.hash + "] - Default template is [" + template + "]";
        this.log({
          error: errStr,
          locale: locale,
          id: id,
          defaultTemplate: template
        });
      }
      if (template) {
        return _.template(template, vars);
      } else {
        return id;
      }
    },
    log: function(error) {
      if (/dev/.test(window.location.host)) {
        if ((typeof console !== "undefined" && console !== null ? console.log : void 0) != null) {
          return console.log(error);
        }
      } else {
        _.extend(error, {
          host: window.location.host,
          hash: window.location.hash
        });
        return this.castor.error(JSON.stringify(error));
      }
    }
  };
}).call(this);
}, "web-lib/mixins/i18n_phone_form": function(exports, require, module) {(function() {
  exports.i18nPhoneForm = {
    _events: {
      'change select[data-country-code-prefix]': 'setCountryCodePrefix'
    },
    setCountryCodePrefix: function(e) {
      var $el, prefix;
      $el = $(e.currentTarget);
      prefix = $el.find('option:selected').attr('data-prefix');
      return $("#" + ($el.attr('data-country-code-prefix'))).text(prefix);
    }
  };
}).call(this);
}, "web-lib/models/country": function(exports, require, module) {(function() {
  var UberModel;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  UberModel = require('web-lib/uber_model').UberModel;
  exports.Country = (function() {
    __extends(Country, UberModel);
    function Country() {
      Country.__super__.constructor.apply(this, arguments);
    }
    Country.prototype.url = function() {
      if (this.id) {
        return "/countries/" + this.id;
      } else {
        return '/countries';
      }
    };
    return Country;
  })();
}).call(this);
}, "web-lib/models/vehicle_type": function(exports, require, module) {(function() {
  var UberModel;
  var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; }, __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  UberModel = require('web-lib/uber_model').UberModel;
  exports.VehicleType = (function() {
    __extends(VehicleType, UberModel);
    function VehicleType() {
      this.toString = __bind(this.toString, this);
      VehicleType.__super__.constructor.apply(this, arguments);
    }
    VehicleType.prototype.endpoint = 'vehicle_types';
    VehicleType.prototype.toTableRow = function(cols) {
      var actions, c, capacity, city_id, columnValues, created_at, created_by_user_id, deleted_at, id, make, minimum_year, model, rows, type, updated_at, updated_by_user_id, _i, _len, _ref;
      id = "<a href='#/vehicle_types/" + (this.get('id')) + "'>" + (this.get('id')) + "</a>";
      if (this.get('created_at')) {
        created_at = app.helpers.formatDate(this.get('created_at'));
      }
      if (this.get('updated_at')) {
        updated_at = app.helpers.formatDate(this.get('updated_at'));
      }
      if (this.get('deleted_at')) {
        deleted_at = app.helpers.formatDate(this.get('deleted_at'));
      }
      created_by_user_id = "<a href='#/clients/" + (this.get('created_by_user_id')) + "'>" + (this.get('created_by_user_id')) + "</a>";
      updated_by_user_id = "<a href='#/clients/" + (this.get('updated_by_user_id')) + "'>" + (this.get('updated_by_user_id')) + "</a>";
      city_id = (_ref = this.get('city')) != null ? _ref.get('display_name') : void 0;
      type = this.get('type');
      make = this.get('make');
      model = this.get('model');
      capacity = this.get('capacity');
      minimum_year = this.get('minimum_year');
      actions = "<a href='#/vehicle_types/" + (this.get('id')) + "'>Show</a>";
      if (!this.get('deleted_at')) {
        actions += " <a href='#/vehicle_types/" + (this.get('id')) + "/edit'>Edit</a>";
        actions += " <a id='" + (this.get('id')) + "' class='delete' href='#'>Delete</a>";
      }
      columnValues = {
        id: id,
        created_at: created_at,
        updated_at: updated_at,
        deleted_at: deleted_at,
        created_by_user_id: created_by_user_id,
        updated_by_user_id: updated_by_user_id,
        city_id: city_id,
        type: type,
        make: make,
        model: model,
        capacity: capacity,
        minimum_year: minimum_year,
        actions: actions
      };
      rows = [];
      for (_i = 0, _len = cols.length; _i < _len; _i++) {
        c = cols[_i];
        rows.push(columnValues[c] ? columnValues[c] : '-');
      }
      return rows;
    };
    VehicleType.prototype.toString = function() {
      return this.get('make') + ' ' + this.get('model') + ' ' + this.get('type') + (" (" + (this.get('capacity')) + ")");
    };
    return VehicleType;
  })();
}).call(this);
}, "web-lib/templates/footer": function(exports, require, module) {module.exports = function(__obj) {
  if (!__obj) __obj = {};
  var __out = [], __capture = function(callback) {
    var out = __out, result;
    __out = [];
    callback.call(this);
    result = __out.join('');
    __out = out;
    return __safe(result);
  }, __sanitize = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else if (typeof value !== 'undefined' && value != null) {
      return __escape(value);
    } else {
      return '';
    }
  }, __safe, __objSafe = __obj.safe, __escape = __obj.escape;
  __safe = __obj.safe = function(value) {
    if (value && value.ecoSafe) {
      return value;
    } else {
      if (!(typeof value !== 'undefined' && value != null)) value = '';
      var result = new String(value);
      result.ecoSafe = true;
      return result;
    }
  };
  if (!__escape) {
    __escape = __obj.escape = function(value) {
      return ('' + value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    };
  }
  (function() {
    (function() {
      var locale, title, _ref;
      __out.push('<div class="footer_col_2">\n  <ul>\n    <li class="head">');
      __out.push(__sanitize(t("Info")));
      __out.push('</li>\n    <li><a href="https://www.uber.com/learn">');
      __out.push(__sanitize(t("Learn More")));
      __out.push('</a></li>\n    <li><a href="https://www.uber.com/learn#pricing">');
      __out.push(__sanitize(t("Pricing")));
      __out.push('</a></li>\n    <li><a href="http://support.uber.com">');
      __out.push(__sanitize(t("Support & FAQ")));
      __out.push('</a></li>\n    <li><a href="https://partners.uber.com/#!/partners/new">');
      __out.push(__sanitize(t("Apply to Drive")));
      __out.push('</a></li>\n  </ul>\n</div>\n<div class="footer_col_2">\n  <ul>\n    <li class="head">');
      __out.push(__sanitize(t("Social")));
      __out.push('</li>\n    <li><a href="http://www.twitter.com/uber">');
      __out.push(__sanitize(t("Twitter")));
      __out.push('</a></li>\n    <li><a href="http://www.facebook.com/uber">');
      __out.push(__sanitize(t("Facebook")));
      __out.push('</a></li>\n    <li><a href="http://blog.uber.com">');
      __out.push(__sanitize(t("Blog")));
      __out.push('</a></li>\n    <li><a href="https://www.uber.com/contact">');
      __out.push(__sanitize(t("Contact Us")));
      __out.push('</a></li>\n  </ul>\n</div>\n<div class="footer_col_2">\n  <ul>\n    <li class="head">');
      __out.push(__sanitize(t("Phones")));
      __out.push('</li>\n    <li><a href="https://www.uber.com/phones/text">');
      __out.push(__sanitize(t("Text Message")));
      __out.push('</a></li>\n    <li><a href="https://www.uber.com/phones/iphone">');
      __out.push(__sanitize(t("iPhone")));
      __out.push('</a></li>\n    <li><a href="https://www.uber.com/phones/android">');
      __out.push(__sanitize(t("Android")));
      __out.push('</a></li>\n  </ul>\n</div>\n<div class="footer_col_2">\n  <ul>\n    <li class="head">');
      __out.push(__sanitize(t("Company_Footer")));
      __out.push('</li>\n    <li><a href="https://www.uber.com/privacy">');
      __out.push(__sanitize(t("Privacy Policy")));
      __out.push('</a></li>\n    <li><a href="https://www.uber.com/terms">');
      __out.push(__sanitize(t("Terms")));
      __out.push('</a></li>\n    <li><a href="https://www.uber.com/jobs">');
      __out.push(__sanitize(t("Jobs")));
      __out.push('</a></li>\n  </ul>\n</div>\n<div class="footer_col_copyright">\n  <p>');
      __out.push(t("Copyright &copy; Uber Technologies, Inc."));
      __out.push('</p>\n    <p>\n      ');
      __out.push(__sanitize(t('Language:')));
      __out.push('\n      ');
      _ref = typeof i18n !== "undefined" && i18n !== null ? i18n.locales : void 0;
      for (locale in _ref) {
        title = _ref[locale];
        __out.push('\n        ');
        if (locale === (typeof i18n !== "undefined" && i18n !== null ? i18n.getLocale() : void 0)) {
          __out.push('\n          <span class="language current_language" id=\'');
          __out.push(__sanitize(locale));
          __out.push('\'>');
          __out.push(__sanitize(title));
          __out.push('</span>\n        ');
        } else {
          __out.push('\n            <a href="');
          __out.push(__sanitize(window.location.href));
          __out.push('" class="language" id=\'');
          __out.push(__sanitize(locale));
          __out.push('\' title="');
          __out.push(__sanitize(title));
          __out.push('">');
          __out.push(__sanitize(title));
          __out.push('</a>\n          ');
        }
        __out.push('\n        ');
      }
      __out.push('\n    </p>\n</div>\n');
    }).call(this);
    
  }).call(__obj);
  __obj.safe = __objSafe, __obj.escape = __escape;
  return __out.join('');
}}, "web-lib/translations/en": function(exports, require, module) {(function() {
  exports.translations = {
    "Info": "Info",
    "Learn More": "Learn More",
    "Pricing": "Pricing",
    "FAQ": "FAQ",
    "Support": "Support",
    "Support & FAQ": "Support & FAQ",
    "Contact Us": "Contact Us",
    "Jobs": "Jobs",
    "Phones": "Phones",
    "Text Message": "Text Message",
    "iPhone": "iPhone",
    "Android": "Android",
    "Drivers": "Drivers",
    "Apply": "Apply",
    "Sign In": "Sign In",
    "Social": "Social",
    "Twitter": "Twitter",
    "Facebook": "Facebook",
    "Blog": "Blog",
    "Legal": "Legal",
    "Company_Footer": "Company",
    "Privacy Policy": "Privacy Policy",
    "Terms": "Terms",
    "Copyright &copy; Uber Technologies, Inc.": "Copyright &copy; Uber Technologies, Inc.",
    "Language:": "Language:",
    "Apply to Drive": "Apply to Drive",
    "Expiration": "Expiration",
    "Fare": "Fare",
    "Driver": "Driver ",
    "Dashboard": "Dashboard",
    "Forgot Password": "Forgot Password",
    "Trip Details": "Trip Details",
    "Save": "Save",
    "Cancel": "Cancel",
    "Edit": "Edit",
    "Password": "Password",
    "First Name": "First Name",
    "Last Name": "Last Name",
    "Email Address": "Email Address",
    "Submit": "Submit",
    "Mobile Number": "Mobile Number",
    "Zip Code": "Zip Code",
    "Sign Out": "Sign Out",
    "Confirm Email Message": "Attempting to confirm email...",
    "Upload": "Upload",
    "Rating": "Rating",
    "Pickup Time": "Pickup Time",
    "2011": "2011",
    "2012": "2012",
    "2013": "2013",
    "2014": "2014",
    "2015": "2015",
    "2016": "2016",
    "2017": "2017",
    "2018": "2018",
    "2019": "2019",
    "2020": "2020",
    "2021": "2021",
    "2022": "2022",
    "01": "01",
    "02": "02",
    "03": "03",
    "04": "04",
    "05": "05",
    "06": "06",
    "07": "07",
    "08": "08",
    "09": "09",
    "10": "10",
    "11": "11",
    "12": "12"
  };
}).call(this);
}, "web-lib/translations/fr": function(exports, require, module) {(function() {
  exports.translations = {
    "Info": "Info",
    "Learn More": "En Savoir Plus",
    "Pricing": "Calcul du Prix",
    "Support & FAQ": "Aide & FAQ",
    "Contact Us": "Contactez Nous",
    "Jobs": "Emplois",
    "Phones": "Téléphones",
    "Text Message": "SMS",
    "iPhone": "iPhone",
    "Android": "Android",
    "Apply to Drive": "Candidature Chauffeur",
    "Sign In": "Connexion",
    "Social": "Contact",
    "Twitter": "Twitter",
    "Facebook": "Facebook",
    "Blog": "Blog",
    "Privacy Policy": "Protection des Données Personelles",
    "Terms": "Conditions Générales",
    "Copyright &copy; Uber Technologies, Inc.": "© Uber, Inc.",
    "Language:": "Langue:",
    "Forgot Password": "Mot de passe oublié",
    "Company_Footer": "À Propos d'Uber",
    "Expiration": "Expiration",
    "Fare": "Tarif",
    "Driver": "Chauffeur",
    "Drivers": "Chauffeurs",
    "Dashboard": "Tableau de bord",
    "Forgot Password": "Mot de passe oublié",
    "Forgot Password?": "Mot de passe oublié?",
    "Trip Details": "Détails de la course",
    "Save": "Enregistrer",
    "Cancel": "Annuler",
    "Edit": "Modifier",
    "Password": "Mot de passe",
    "First Name": "Prénom",
    "Last Name": "Nom",
    "Email Address": "E-mail",
    "Submit": "Soumettre",
    "Mobile Number": "Téléphone Portable",
    "Zip Code": "Code Postal",
    "Sign Out": "Se déconnecter",
    "Confirm Email Message": "E-mail de confirmation",
    "Upload": "Télécharger",
    "Rating": "Notation",
    "Pickup Time": "Heure de prise en charge",
    "2011": "2011",
    "2012": "2012",
    "2013": "2013",
    "2014": "2014",
    "2015": "2015",
    "2016": "2016",
    "2017": "2017",
    "2018": "2018",
    "2019": "2019",
    "2020": "2020",
    "2021": "2021",
    "2022": "2022",
    "01": "01",
    "02": "02",
    "03": "03",
    "04": "04",
    "05": "05",
    "06": "06",
    "07": "07",
    "08": "08",
    "09": "09",
    "10": "10",
    "11": "11",
    "12": "12"
  };
}).call(this);
}, "web-lib/uber_collection": function(exports, require, module) {(function() {
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.UberCollection = (function() {
    __extends(UberCollection, Backbone.Collection);
    function UberCollection() {
      UberCollection.__super__.constructor.apply(this, arguments);
    }
    UberCollection.prototype.parse = function(data) {
      var model, tmp, _i, _in, _len, _out;
      _in = data.resources || data;
      _out = [];
      if (data.meta) {
        this.meta = data.meta;
      }
      for (_i = 0, _len = _in.length; _i < _len; _i++) {
        model = _in[_i];
        tmp = new this.model;
        tmp.set(tmp.parse(model));
        _out.push(tmp);
      }
      return _out;
    };
    UberCollection.prototype.isRenderable = function() {
      if (this.models.length) {
        return true;
      }
    };
    UberCollection.prototype.toTableRows = function(cols) {
      var tableRows;
      tableRows = [];
      _.each(this.models, function(model) {
        return tableRows.push(model.toTableRow(cols));
      });
      return tableRows;
    };
    return UberCollection;
  })();
}).call(this);
}, "web-lib/uber_model": function(exports, require, module) {(function() {
  var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; }, __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  }, __indexOf = Array.prototype.indexOf || function(item) {
    for (var i = 0, l = this.length; i < l; i++) {
      if (this[i] === item) return i;
    }
    return -1;
  };
  exports.UberModel = (function() {
    __extends(UberModel, Backbone.Model);
    function UberModel() {
      this.refetch = __bind(this.refetch, this);
      this.fetch = __bind(this.fetch, this);
      this.save = __bind(this.save, this);
      this.parse = __bind(this.parse, this);
      UberModel.__super__.constructor.apply(this, arguments);
    }
    UberModel.prototype.endpoint = 'set_api_endpoint_in_subclass';
    UberModel.prototype.refetchOptions = {};
    UberModel.prototype.url = function(type) {
      var endpoint_path;
      endpoint_path = "/" + this.endpoint;
      if (this.get('id')) {
        return endpoint_path + ("/" + (this.get('id')));
      } else {
        return endpoint_path;
      }
    };
    UberModel.prototype.isRenderable = function() {
      var i, key, value, _ref;
      i = 0;
      _ref = this.attributes;
      for (key in _ref) {
        if (!__hasProp.call(_ref, key)) continue;
        value = _ref[key];
        if (this.attributes.hasOwnProperty(key)) {
          i += 1;
        }
        if (i > 1) {
          return true;
        }
      }
      return !(i === 1);
    };
    UberModel.prototype.parse = function(response) {
      var attrs, key, model, models, _i, _j, _k, _len, _len2, _len3, _ref, _ref2;
      if (typeof response === 'object') {
        _ref = _.intersection(_.keys(app.models), _.keys(response));
        for (_i = 0, _len = _ref.length; _i < _len; _i++) {
          key = _ref[_i];
          if (response[key]) {
            attrs = this.parse(response[key]);
            if (typeof attrs === 'object') {
              response[key] = new app.models[key](attrs);
            }
          }
        }
        _ref2 = _.intersection(_.keys(app.collections), _.keys(response));
        for (_j = 0, _len2 = _ref2.length; _j < _len2; _j++) {
          key = _ref2[_j];
          models = response[key];
          if (_.isArray(models)) {
            response[key] = new app.collections[key];
            for (_k = 0, _len3 = models.length; _k < _len3; _k++) {
              model = models[_k];
              attrs = app.collections[key].prototype.model.prototype.parse(model);
              response[key].add(new response[key].model(attrs));
            }
          }
        }
      }
      return response;
    };
    UberModel.prototype.save = function(attributes, options) {
      var attr, _i, _j, _len, _len2, _ref, _ref2;
      if (options == null) {
        options = {};
      }
      _ref = _.intersection(_.keys(app.models), _.keys(this.attributes));
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        attr = _ref[_i];
        if (typeof this.get(attr) === "object") {
          this.unset(attr, {
            silent: true
          });
        }
      }
      _ref2 = _.intersection(_.keys(app.collections), _.keys(this.attributes));
      for (_j = 0, _len2 = _ref2.length; _j < _len2; _j++) {
        attr = _ref2[_j];
        if (typeof this.get(attr) === "object") {
          this.unset(attr, {
            silent: true
          });
        }
      }
      if ((options != null) && options.diff && (attributes != null) && attributes !== {}) {
        attributes['id'] = this.get('id');
        attributes['token'] = this.get('token');
        this.clear({
          'silent': true
        });
        this.set(attributes, {
          silent: true
        });
      }
      if (__indexOf.call(_.keys(options), "data") < 0 && __indexOf.call(_.keys(this.refetchOptions || {}), "data") >= 0) {
        options.data = this.refetchOptions.data;
      }
      return Backbone.Model.prototype.save.call(this, attributes, options);
    };
    UberModel.prototype.fetch = function(options) {
      this.refetchOptions = options;
      return Backbone.Model.prototype.fetch.call(this, options);
    };
    UberModel.prototype.refetch = function() {
      return this.fetch(this.refetchOptions);
    };
    return UberModel;
  })();
}).call(this);
}, "web-lib/uber_router": function(exports, require, module) {(function() {
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.UberRouter = (function() {
    __extends(UberRouter, Backbone.Router);
    function UberRouter() {
      UberRouter.__super__.constructor.apply(this, arguments);
    }
    UberRouter.prototype.datePickers = function(format) {
      if (format == null) {
        format = "%Z-%m-%dT%H:%i:%s%:";
      }
      $('.datepicker').AnyTime_noPicker();
      return $('.datepicker').AnyTime_picker({
        'format': format,
        'formatUtcOffset': '%@'
      });
    };
    UberRouter.prototype.autoGrowInput = function() {
      return $('.editable input').autoGrowInput();
    };
    UberRouter.prototype.windowTitle = function(title) {
      return $(document).attr('title', title);
    };
    return UberRouter;
  })();
}).call(this);
}, "web-lib/uber_show_view": function(exports, require, module) {(function() {
  var UberView;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  }, __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; };
  UberView = require('web-lib/uber_view').UberView;
  exports.UberShowView = (function() {
    __extends(UberShowView, UberView);
    function UberShowView() {
      UberShowView.__super__.constructor.apply(this, arguments);
    }
    UberShowView.prototype.view = 'show';
    UberShowView.prototype.events = {
      'click #edit': 'edit',
      'submit form': 'save',
      'click .cancel': 'cancel'
    };
    UberShowView.prototype.errors = null;
    UberShowView.prototype.showTemplate = null;
    UberShowView.prototype.editTemplate = null;
    UberShowView.prototype.initialize = function() {
      if (this.init_hook) {
        this.init_hook();
      }
      _.bindAll(this, 'render');
      return this.model.bind('change', this.render);
    };
    UberShowView.prototype.render = function() {
      var $el;
      $el = $(this.el);
      this.selectView();
      if (this.view === 'show') {
        $el.html(this.showTemplate({
          model: this.model
        }));
      } else if (this.view === 'edit') {
        $el.html(this.editTemplate({
          model: this.model,
          errors: this.errors || {},
          collections: this.collections || {}
        }));
      } else {
        $el.html(this.newTemplate({
          model: this.model,
          errors: this.errors || {},
          collections: this.collections || {}
        }));
      }
      if (this.render_hook) {
        this.render_hook();
      }
      this.errors = null;
      this.userIdsToLinkedNames();
      this.datePickers();
      return this.place();
    };
    UberShowView.prototype.selectView = function() {
      var url;
      if (this.options.urlRendering) {
        url = window.location.hash;
        if (url.match(/\/new/)) {
          return this.view = 'new';
        } else if (url.match(/\/edit/)) {
          return this.view = 'edit';
        } else {
          return this.view = 'show';
        }
      }
    };
    UberShowView.prototype.edit = function(e) {
      e.preventDefault();
      if (this.options.urlRendering) {
        window.location.hash = '#/' + this.model.endpoint + '/' + this.model.get('id') + '/edit';
      } else {
        this.view = 'edit';
      }
      return this.model.change();
    };
    UberShowView.prototype.save = function(e) {
      var attributes, ele, form_attrs, _i, _len, _ref;
      e.preventDefault();
      attributes = $(e.currentTarget).serializeToJson();
      form_attrs = {};
      _ref = $('input[type="radio"]');
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        ele = _ref[_i];
        if ($(ele).is(':checked')) {
          form_attrs[$(ele).attr('name')] = $(ele).attr('value');
        }
      }
      attributes = _.extend(attributes, form_attrs);
      if (this.relationships) {
        attributes = _.extend(attributes, {
          relationships: this.relationships
        });
      }
      if (this.filter_attributes != null) {
        this.filter_attributes(attributes);
      }
      return this.model.save(attributes, {
        silent: true,
        success: __bind(function(model) {
          if (this.options.urlRendering) {
            window.location.hash = '#/' + this.model.endpoint + '/' + this.model.get('id');
          } else {
            this.view = 'show';
          }
          return this.flash('success', "Uber save!");
        }, this),
        statusCode: {
          406: __bind(function(xhr) {
            this.errors = JSON.parse(xhr.responseText);
            return this.flash('error', 'That was not Uber.');
          }, this)
        },
        error: __bind(function(model, xhr) {
          var code, message, responseJSON, responseText;
          code = xhr.status;
          responseText = xhr.responseText;
          if (responseText) {
            responseJSON = JSON.parse(responseText);
          }
          if (responseJSON && (typeof responseJSON === 'object') && (responseJSON.hasOwnProperty('error'))) {
            message = responseJSON.error;
          }
          return this.flash('error', (code || 'Unknown') + ' error' + (': ' + message || ''));
        }, this),
        complete: __bind(function() {
          return this.model.change();
        }, this)
      });
    };
    UberShowView.prototype.cancel = function(e) {
      e.preventDefault();
      if (this.options.urlRendering) {
        window.location.hash = '#/' + this.model.endpoint + '/' + this.model.get('id');
      } else {
        this.view = 'show';
      }
      return this.model.fetch({
        silent: true,
        complete: __bind(function() {
          return this.model.change();
        }, this)
      });
    };
    return UberShowView;
  })();
}).call(this);
}, "web-lib/uber_sync": function(exports, require, module) {(function() {
  var methodType;
  var __indexOf = Array.prototype.indexOf || function(item) {
    for (var i = 0, l = this.length; i < l; i++) {
      if (this[i] === item) return i;
    }
    return -1;
  };
  methodType = {
    create: 'POST',
    update: 'PUT',
    "delete": 'DELETE',
    read: 'GET'
  };
  exports.UberSync = function(method, model, options) {
    var token;
    options.type = methodType[method];
    options.url = _.isString(this.url) ? '/api' + this.url : '/api' + this.url(options.type);
    options.data = _.extend({}, options.data);
    if (__indexOf.call(_.keys(options.data), "city_id") < 0) {
      if ($.cookie('city_filter')) {
        _.extend(options.data, {
          city_id: $.cookie('city_filter')
        });
      }
    } else {
      delete options.data['city_id'];
    }
    if (options.type === 'POST' || options.type === 'PUT') {
      _.extend(options.data, model.toJSON());
    }
    token = $.cookie('token') ? $.cookie('token') : typeof USER !== "undefined" && USER !== null ? USER.get('token') : "";
    _.extend(options.data, {
      token: token
    });
    if (method === "delete") {
      options.contentType = 'application/json';
      options.data = JSON.stringify(options.data);
    }
    return $.ajax(options);
  };
}).call(this);
}, "web-lib/uber_view": function(exports, require, module) {(function() {
  var __bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; }, __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  exports.UberView = (function() {
    __extends(UberView, Backbone.View);
    function UberView() {
      this.processDocumentUpload = __bind(this.processDocumentUpload, this);
      UberView.__super__.constructor.apply(this, arguments);
    }
    UberView.prototype.className = 'view_container';
    UberView.prototype.hashId = function() {
      return parseInt(location.hash.split('/')[2]);
    };
    UberView.prototype.place = function(content) {
      var $target;
      $target = this.options.scope ? this.options.scope.find(this.options.selector) : $(this.options.selector);
      $target[this.options.method || 'html'](content || this.el);
      this.delegateEvents();
      $('#spinner').hide();
      return this;
    };
    UberView.prototype.mixin = function(m, args) {
      var events, self;
      if (args == null) {
        args = {};
      }
      self = this;
      events = m._events;
      _.extend(this, m);
      if (m.initialize) {
        m.initialize(self, args);
      }
      return _.each(_.keys(events), function(key) {
        var event, func, selector, split;
        split = key.split(' ');
        event = split[0];
        selector = split[1];
        func = events[key];
        return $(self.el).find(selector).live(event, function(e) {
          return self[func](e);
        });
      });
    };
    UberView.prototype.datePickers = function(format) {
      if (format == null) {
        format = "%Z-%m-%dT%H:%i:%s%:";
      }
      $('.datepicker').AnyTime_noPicker();
      return $('.datepicker').AnyTime_picker({
        'format': format,
        'formatUtcOffset': '%@'
      });
    };
    UberView.prototype.dataTable = function(collection, selector, options, params, cols) {
      var defaults;
      if (selector == null) {
        selector = 'table';
      }
      if (options == null) {
        options = {};
      }
      if (params == null) {
        params = {};
      }
      if (cols == null) {
        cols = [];
      }
      $(selector).empty();
      if (!cols.length) {
        cols = collection.defaultColumns;
      }
      defaults = {
        aoColumns: collection.tableColumns(cols),
        bDestroy: true,
        bSort: false,
        bProcessing: true,
        bFilter: false,
        bServerSide: true,
        bPaginate: true,
        bScrollInfinite: true,
        bScrollCollapse: true,
        sScrollY: '600px',
        iDisplayLength: 50,
        fnServerData: function(source, data, callback) {
          var defaultParams;
          defaultParams = {
            limit: data[4].value,
            offset: data[3].value
          };
          return collection.fetch({
            data: _.extend(defaultParams, params),
            success: function() {
              return callback({
                aaData: collection.toTableRows(cols),
                iTotalRecords: collection.meta.count,
                iTotalDisplayRecords: collection.meta.count
              });
            },
            error: function() {
              return new Error({
                message: 'Loading error.'
              });
            }
          });
        },
        fnRowCallback: function(nRow, aData, iDisplayIndex, iDisplayIndexFull) {
          $('[data-tooltip]', nRow).qtip({
            content: {
              attr: 'data-tooltip'
            },
            style: {
              classes: "ui-tooltip-light ui-tooltip-rounded ui-tooltip-shadow"
            }
          });
          return nRow;
        }
      };
      return $(this.el).find(selector).dataTable(_.extend(defaults, options));
    };
    UberView.prototype.dataTableLocal = function(collection, selector, options, params, cols) {
      var $dataTable, defaults;
      if (selector == null) {
        selector = 'table';
      }
      if (options == null) {
        options = {};
      }
      if (params == null) {
        params = {};
      }
      if (cols == null) {
        cols = [];
      }
      $(selector).empty();
      if (!cols.length || cols.length === 0) {
        cols = collection.defaultColumns;
      }
      defaults = {
        aaData: collection.toTableRows(cols),
        aoColumns: collection.tableColumns(cols),
        bDestroy: true,
        bSort: false,
        bProcessing: true,
        bFilter: false,
        bScrollInfinite: true,
        bScrollCollapse: true,
        sScrollY: '600px',
        iDisplayLength: -1
      };
      $dataTable = $(this.el).find(selector).dataTable(_.extend(defaults, options));
      _.delay(__bind(function() {
        if ($dataTable && $dataTable.length > 0) {
          return $dataTable.fnAdjustColumnSizing();
        }
      }, this), 1);
      return $dataTable;
    };
    UberView.prototype.reverseGeocode = function() {
      var $el;
      return '';
      $el = $(this.el);
      return this.requireMaps(function() {
        var geocoder;
        geocoder = new google.maps.Geocoder();
        return $el.find('[data-point]').each(function() {
          var $this, latLng, point;
          $this = $(this);
          point = JSON.parse($this.attr('data-point'));
          latLng = new google.maps.LatLng(point.latitude, point.longitude);
          return geocoder.geocode({
            latLng: latLng
          }, function(data, status) {
            if (status === google.maps.GeocoderStatus.OK) {
              return $this.text(data[0].formatted_address);
            }
          });
        });
      });
    };
    UberView.prototype.userIdsToLinkedNames = function() {
      var $el;
      $el = $(this.el);
      return $el.find('a[data-user-id][data-user-type]').each(function() {
        var $this, user, userType;
        $this = $(this);
        userType = $this.attr('data-user-type') === 'user' ? 'client' : $this.attr('data-user-type');
        user = new app.models[userType]({
          id: $this.attr('data-user-id')
        });
        return user.fetch({
          success: function(user) {
            return $this.html(app.helpers.linkedName(user)).attr('href', "!/" + user.role + "s/" + user.id);
          },
          error: function() {
            if ($this.attr('data-user-type') === 'user') {
              user = new app.models['driver']({
                id: $this.attr('data-user-id')
              });
              return user.fetch({
                success: function(user) {
                  return $this.html(app.helpers.linkedName(user)).attr('href', "!/driver/" + user.id);
                }
              });
            }
          }
        });
      });
    };
    UberView.prototype.selectedCity = function() {
      var $selected, city, cityFilter;
      cityFilter = $.cookie('city_filter');
      $selected = $("#city_filter option[value=" + cityFilter + "]");
      if (city_filter && $selected.length) {
        return city = {
          lat: parseFloat($selected.attr('data-lat')),
          lng: parseFloat($selected.attr('data-lng')),
          timezone: $selected.attr('data-timezone')
        };
      } else {
        return city = {
          lat: 37.775,
          lng: -122.45,
          timezone: 'Etc/UTC'
        };
      }
    };
    UberView.prototype.updateModel = function(e, success) {
      var $el, attrs, model, self;
      e.preventDefault();
      $el = $(e.currentTarget);
      self = this;
      model = new this.model.__proto__.constructor({
        id: this.model.id
      });
      attrs = {};
      $el.find('[name]').each(function() {
        var $this;
        $this = $(this);
        return attrs["" + ($this.attr('name'))] = $this.val();
      });
      self.model.set(attrs);
      $el.find('span.error').text('');
      return model.save(attrs, {
        complete: function(xhr) {
          var response;
          response = JSON.parse(xhr.responseText);
          switch (xhr.status) {
            case 200:
              self.model = model;
              $el.find('[name]').val('');
              if (success) {
                return success();
              }
              break;
            case 406:
              return _.each(response, function(error, field) {
                return $el.find("[name=" + field + "]").parent().find('span.error').text(error);
              });
            default:
              return this.unanticipatedError(response);
          }
        }
      });
    };
    UberView.prototype.autoUpdateModel = function(e) {
      var $el, arg, model, self, val;
      $el = $(e.currentTarget);
      val = $el.val();
      self = this;
      if (val !== this.model.get($el.attr('id'))) {
        arg = {};
        arg[$el.attr('id')] = $el.is(':checkbox') ? $el.is(':checked') ? 1 : 0 : val;
        $('.editable span').empty();
        this.model.set(arg);
        model = new this.model.__proto__.constructor({
          id: this.model.id
        });
        return model.save(arg, {
          complete: function(xhr) {
            var key, response, _i, _len, _ref, _results;
            response = JSON.parse(xhr.responseText);
            switch (xhr.status) {
              case 200:
                self.flash('success', 'Saved!');
                return $el.blur();
              case 406:
                self.flash('error', 'That was not Uber.');
                _ref = _.keys(response);
                _results = [];
                for (_i = 0, _len = _ref.length; _i < _len; _i++) {
                  key = _ref[_i];
                  _results.push($el.parent().find('span').html(response[key]));
                }
                return _results;
                break;
              default:
                return self.unanticipatedError;
            }
          }
        });
      }
    };
    UberView.prototype.unanticipatedError = function(response) {
      return self.flash('error', response);
    };
    UberView.prototype.flash = function(type, text) {
      var $banner;
      $banner = $("." + type);
      $banner.find('p').text(text).end().css('border', '1px solid #999').animate({
        top: 0
      }, 500);
      return setTimeout(function() {
        return $banner.animate({
          top: -$banner.outerHeight()
        }, 500);
      }, 3000);
    };
    UberView.prototype.requireMaps = function(callback) {
      if (typeof google !== 'undefined' && google.maps) {
        return callback();
      } else {
        return $.getScript("https://www.google.com/jsapi?key=" + CONFIG.googleJsApiKey, function() {
          return google.load('maps', 3, {
            callback: callback,
            other_params: 'sensor=false&language=en'
          });
        });
      }
    };
    UberView.prototype.select_drop_down = function(model, key) {
      var value;
      value = model.get(key);
      if (value) {
        return $("select[id='" + key + "'] option[value='" + value + "']").attr('selected', 'selected');
      }
    };
    UberView.prototype.processDocumentUpload = function(e) {
      var $fi, $form, arbData, curDate, data, expDate, expM, expY, expiration, fileElementId, invalid;
      e.preventDefault();
      $form = $(e.currentTarget);
      $fi = $("input[type=file]", $form);
      $(".validationError").removeClass("validationError");
      if (!$fi.val()) {
        return $fi.addClass("validationError");
      } else {
        fileElementId = $fi.attr('id');
        expY = $("select[name=expiration-year]", $form).val();
        expM = $("select[name=expiration-month]", $form).val();
        invalid = false;
        if (expY && expM) {
          expDate = new Date(expY, expM, 28);
          curDate = new Date();
          if (expDate < curDate) {
            invalid = true;
            $(".expiration", $form).addClass("validationError");
          }
          expiration = "" + expY + "-" + expM + "-28T23:59:59Z";
        }
        arbData = {};
        $(".arbitraryField", $form).each(__bind(function(i, e) {
          arbData[$(e).attr('name')] = $(e).val();
          if ($(e).val() === "") {
            invalid = true;
            return $(e).addClass("validationError");
          }
        }, this));
        if (!invalid) {
          data = {
            token: $.cookie('token') || USER.get('token'),
            name: $("input[name=fileName]", $form).val(),
            meta: escape(JSON.stringify(arbData)),
            user_id: $("input[name=driver_id]", $form).val(),
            vehicle_id: $("input[name=vehicle_id]", $form).val()
          };
          if (expiration) {
            data['expiration'] = expiration;
          }
          $("#spinner").show();
          return $.ajaxFileUpload({
            url: '/api/documents',
            secureuri: false,
            fileElementId: fileElementId,
            data: data,
            complete: __bind(function(resp, status) {
              var key, _i, _len, _ref, _results;
              $("#spinner").hide();
              if (status === "success") {
                if (this.model) {
                  this.model.refetch();
                } else {
                  USER.refetch();
                }
              }
              if (status === "error") {
                _ref = _.keys(resp);
                _results = [];
                for (_i = 0, _len = _ref.length; _i < _len; _i++) {
                  key = _ref[_i];
                  _results.push($("*[name=" + key + "]", $form).addClass("validationError"));
                }
                return _results;
              }
            }, this)
          });
        }
      }
    };
    return UberView;
  })();
}).call(this);
}, "web-lib/views/footer": function(exports, require, module) {(function() {
  var footerTemplate;
  var __hasProp = Object.prototype.hasOwnProperty, __extends = function(child, parent) {
    for (var key in parent) { if (__hasProp.call(parent, key)) child[key] = parent[key]; }
    function ctor() { this.constructor = child; }
    ctor.prototype = parent.prototype;
    child.prototype = new ctor;
    child.__super__ = parent.prototype;
    return child;
  };
  footerTemplate = require('web-lib/templates/footer');
  exports.SharedFooterView = (function() {
    __extends(SharedFooterView, Backbone.View);
    function SharedFooterView() {
      SharedFooterView.__super__.constructor.apply(this, arguments);
    }
    SharedFooterView.prototype.id = 'footer_view';
    SharedFooterView.prototype.events = {
      'click .language': 'intl_set_cookie_locale'
    };
    SharedFooterView.prototype.render = function() {
      $(this.el).html(footerTemplate());
      this.delegateEvents();
      return this;
    };
    SharedFooterView.prototype.intl_set_cookie_locale = function(e) {
      var _ref;
      i18n.setLocale(e != null ? (_ref = e.srcElement) != null ? _ref.id : void 0 : void 0);
      return location.reload();
    };
    return SharedFooterView;
  })();
}).call(this);
}});

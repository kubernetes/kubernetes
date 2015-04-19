/*
 * Copyright 2014 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Generated from http://github.com/GoogleChrome/accessibility-developer-tools/tree/3d4893b4ecd0eb8f4765e04479213d04b240f3e0
 *
 * See project README for build steps.
 */
 
// AUTO-GENERATED CONTENT BELOW: DO NOT EDIT! See above for details.

var COMPILED = !0, goog = goog || {};
goog.global = this;
goog.isDef = function(a) {
  return void 0 !== a;
};
goog.exportPath_ = function(a, b, c) {
  a = a.split(".");
  c = c || goog.global;
  a[0] in c || !c.execScript || c.execScript("var " + a[0]);
  for (var d;a.length && (d = a.shift());) {
    !a.length && goog.isDef(b) ? c[d] = b : c = c[d] ? c[d] : c[d] = {};
  }
};
goog.define = function(a, b) {
  var c = b;
  COMPILED || (goog.global.CLOSURE_UNCOMPILED_DEFINES && Object.prototype.hasOwnProperty.call(goog.global.CLOSURE_UNCOMPILED_DEFINES, a) ? c = goog.global.CLOSURE_UNCOMPILED_DEFINES[a] : goog.global.CLOSURE_DEFINES && Object.prototype.hasOwnProperty.call(goog.global.CLOSURE_DEFINES, a) && (c = goog.global.CLOSURE_DEFINES[a]));
  goog.exportPath_(a, c);
};
goog.DEBUG = !0;
goog.LOCALE = "en";
goog.TRUSTED_SITE = !0;
goog.STRICT_MODE_COMPATIBLE = !1;
goog.provide = function(a) {
  if (!COMPILED) {
    if (goog.isProvided_(a)) {
      throw Error('Namespace "' + a + '" already declared.');
    }
    delete goog.implicitNamespaces_[a];
    for (var b = a;(b = b.substring(0, b.lastIndexOf("."))) && !goog.getObjectByName(b);) {
      goog.implicitNamespaces_[b] = !0;
    }
  }
  goog.exportPath_(a);
};
goog.setTestOnly = function(a) {
  if (COMPILED && !goog.DEBUG) {
    throw a = a || "", Error("Importing test-only code into non-debug environment" + a ? ": " + a : ".");
  }
};
goog.forwardDeclare = function(a) {
};
COMPILED || (goog.isProvided_ = function(a) {
  return!goog.implicitNamespaces_[a] && goog.isDefAndNotNull(goog.getObjectByName(a));
}, goog.implicitNamespaces_ = {});
goog.getObjectByName = function(a, b) {
  for (var c = a.split("."), d = b || goog.global, e;e = c.shift();) {
    if (goog.isDefAndNotNull(d[e])) {
      d = d[e];
    } else {
      return null;
    }
  }
  return d;
};
goog.globalize = function(a, b) {
  var c = b || goog.global, d;
  for (d in a) {
    c[d] = a[d];
  }
};
goog.addDependency = function(a, b, c) {
  if (goog.DEPENDENCIES_ENABLED) {
    var d;
    a = a.replace(/\\/g, "/");
    for (var e = goog.dependencies_, f = 0;d = b[f];f++) {
      e.nameToPath[d] = a, a in e.pathToNames || (e.pathToNames[a] = {}), e.pathToNames[a][d] = !0;
    }
    for (d = 0;b = c[d];d++) {
      a in e.requires || (e.requires[a] = {}), e.requires[a][b] = !0;
    }
  }
};
goog.ENABLE_DEBUG_LOADER = !0;
goog.require = function(a) {
  if (!COMPILED && !goog.isProvided_(a)) {
    if (goog.ENABLE_DEBUG_LOADER) {
      var b = goog.getPathFromDeps_(a);
      if (b) {
        goog.included_[b] = !0;
        goog.writeScripts_();
        return;
      }
    }
    a = "goog.require could not find: " + a;
    goog.global.console && goog.global.console.error(a);
    throw Error(a);
  }
};
goog.basePath = "";
goog.nullFunction = function() {
};
goog.identityFunction = function(a, b) {
  return a;
};
goog.abstractMethod = function() {
  throw Error("unimplemented abstract method");
};
goog.addSingletonGetter = function(a) {
  a.getInstance = function() {
    if (a.instance_) {
      return a.instance_;
    }
    goog.DEBUG && (goog.instantiatedSingletons_[goog.instantiatedSingletons_.length] = a);
    return a.instance_ = new a;
  };
};
goog.instantiatedSingletons_ = [];
goog.DEPENDENCIES_ENABLED = !COMPILED && goog.ENABLE_DEBUG_LOADER;
goog.DEPENDENCIES_ENABLED && (goog.included_ = {}, goog.dependencies_ = {pathToNames:{}, nameToPath:{}, requires:{}, visited:{}, written:{}}, goog.inHtmlDocument_ = function() {
  var a = goog.global.document;
  return "undefined" != typeof a && "write" in a;
}, goog.findBasePath_ = function() {
  if (goog.global.CLOSURE_BASE_PATH) {
    goog.basePath = goog.global.CLOSURE_BASE_PATH;
  } else {
    if (goog.inHtmlDocument_()) {
      for (var a = goog.global.document.getElementsByTagName("script"), b = a.length - 1;0 <= b;--b) {
        var c = a[b].src, d = c.lastIndexOf("?"), d = -1 == d ? c.length : d;
        if ("base.js" == c.substr(d - 7, 7)) {
          goog.basePath = c.substr(0, d - 7);
          break;
        }
      }
    }
  }
}, goog.importScript_ = function(a) {
  var b = goog.global.CLOSURE_IMPORT_SCRIPT || goog.writeScriptTag_;
  !goog.dependencies_.written[a] && b(a) && (goog.dependencies_.written[a] = !0);
}, goog.writeScriptTag_ = function(a) {
  if (goog.inHtmlDocument_()) {
    var b = goog.global.document;
    if ("complete" == b.readyState) {
      if (/\bdeps.js$/.test(a)) {
        return!1;
      }
      throw Error('Cannot write "' + a + '" after document load');
    }
    b.write('<script type="text/javascript" src="' + a + '">\x3c/script>');
    return!0;
  }
  return!1;
}, goog.writeScripts_ = function() {
  function a(e) {
    if (!(e in d.written)) {
      if (!(e in d.visited) && (d.visited[e] = !0, e in d.requires)) {
        for (var g in d.requires[e]) {
          if (!goog.isProvided_(g)) {
            if (g in d.nameToPath) {
              a(d.nameToPath[g]);
            } else {
              throw Error("Undefined nameToPath for " + g);
            }
          }
        }
      }
      e in c || (c[e] = !0, b.push(e));
    }
  }
  var b = [], c = {}, d = goog.dependencies_, e;
  for (e in goog.included_) {
    d.written[e] || a(e);
  }
  for (e = 0;e < b.length;e++) {
    if (b[e]) {
      goog.importScript_(goog.basePath + b[e]);
    } else {
      throw Error("Undefined script input");
    }
  }
}, goog.getPathFromDeps_ = function(a) {
  return a in goog.dependencies_.nameToPath ? goog.dependencies_.nameToPath[a] : null;
}, goog.findBasePath_(), goog.global.CLOSURE_NO_DEPS || goog.importScript_(goog.basePath + "deps.js"));
goog.typeOf = function(a) {
  var b = typeof a;
  if ("object" == b) {
    if (a) {
      if (a instanceof Array) {
        return "array";
      }
      if (a instanceof Object) {
        return b;
      }
      var c = Object.prototype.toString.call(a);
      if ("[object Window]" == c) {
        return "object";
      }
      if ("[object Array]" == c || "number" == typeof a.length && "undefined" != typeof a.splice && "undefined" != typeof a.propertyIsEnumerable && !a.propertyIsEnumerable("splice")) {
        return "array";
      }
      if ("[object Function]" == c || "undefined" != typeof a.call && "undefined" != typeof a.propertyIsEnumerable && !a.propertyIsEnumerable("call")) {
        return "function";
      }
    } else {
      return "null";
    }
  } else {
    if ("function" == b && "undefined" == typeof a.call) {
      return "object";
    }
  }
  return b;
};
goog.isNull = function(a) {
  return null === a;
};
goog.isDefAndNotNull = function(a) {
  return null != a;
};
goog.isArray = function(a) {
  return "array" == goog.typeOf(a);
};
goog.isArrayLike = function(a) {
  var b = goog.typeOf(a);
  return "array" == b || "object" == b && "number" == typeof a.length;
};
goog.isDateLike = function(a) {
  return goog.isObject(a) && "function" == typeof a.getFullYear;
};
goog.isString = function(a) {
  return "string" == typeof a;
};
goog.isBoolean = function(a) {
  return "boolean" == typeof a;
};
goog.isNumber = function(a) {
  return "number" == typeof a;
};
goog.isFunction = function(a) {
  return "function" == goog.typeOf(a);
};
goog.isObject = function(a) {
  var b = typeof a;
  return "object" == b && null != a || "function" == b;
};
goog.getUid = function(a) {
  return a[goog.UID_PROPERTY_] || (a[goog.UID_PROPERTY_] = ++goog.uidCounter_);
};
goog.hasUid = function(a) {
  return!!a[goog.UID_PROPERTY_];
};
goog.removeUid = function(a) {
  "removeAttribute" in a && a.removeAttribute(goog.UID_PROPERTY_);
  try {
    delete a[goog.UID_PROPERTY_];
  } catch (b) {
  }
};
goog.UID_PROPERTY_ = "closure_uid_" + (1E9 * Math.random() >>> 0);
goog.uidCounter_ = 0;
goog.getHashCode = goog.getUid;
goog.removeHashCode = goog.removeUid;
goog.cloneObject = function(a) {
  var b = goog.typeOf(a);
  if ("object" == b || "array" == b) {
    if (a.clone) {
      return a.clone();
    }
    var b = "array" == b ? [] : {}, c;
    for (c in a) {
      b[c] = goog.cloneObject(a[c]);
    }
    return b;
  }
  return a;
};
goog.bindNative_ = function(a, b, c) {
  return a.call.apply(a.bind, arguments);
};
goog.bindJs_ = function(a, b, c) {
  if (!a) {
    throw Error();
  }
  if (2 < arguments.length) {
    var d = Array.prototype.slice.call(arguments, 2);
    return function() {
      var c = Array.prototype.slice.call(arguments);
      Array.prototype.unshift.apply(c, d);
      return a.apply(b, c);
    };
  }
  return function() {
    return a.apply(b, arguments);
  };
};
goog.bind = function(a, b, c) {
  Function.prototype.bind && -1 != Function.prototype.bind.toString().indexOf("native code") ? goog.bind = goog.bindNative_ : goog.bind = goog.bindJs_;
  return goog.bind.apply(null, arguments);
};
goog.partial = function(a, b) {
  var c = Array.prototype.slice.call(arguments, 1);
  return function() {
    var b = c.slice();
    b.push.apply(b, arguments);
    return a.apply(this, b);
  };
};
goog.mixin = function(a, b) {
  for (var c in b) {
    a[c] = b[c];
  }
};
goog.now = goog.TRUSTED_SITE && Date.now || function() {
  return+new Date;
};
goog.globalEval = function(a) {
  if (goog.global.execScript) {
    goog.global.execScript(a, "JavaScript");
  } else {
    if (goog.global.eval) {
      if (null == goog.evalWorksForGlobals_ && (goog.global.eval("var _et_ = 1;"), "undefined" != typeof goog.global._et_ ? (delete goog.global._et_, goog.evalWorksForGlobals_ = !0) : goog.evalWorksForGlobals_ = !1), goog.evalWorksForGlobals_) {
        goog.global.eval(a);
      } else {
        var b = goog.global.document, c = b.createElement("script");
        c.type = "text/javascript";
        c.defer = !1;
        c.appendChild(b.createTextNode(a));
        b.body.appendChild(c);
        b.body.removeChild(c);
      }
    } else {
      throw Error("goog.globalEval not available");
    }
  }
};
goog.evalWorksForGlobals_ = null;
goog.getCssName = function(a, b) {
  var c = function(a) {
    return goog.cssNameMapping_[a] || a;
  }, d = function(a) {
    a = a.split("-");
    for (var b = [], d = 0;d < a.length;d++) {
      b.push(c(a[d]));
    }
    return b.join("-");
  }, d = goog.cssNameMapping_ ? "BY_WHOLE" == goog.cssNameMappingStyle_ ? c : d : function(a) {
    return a;
  };
  return b ? a + "-" + d(b) : d(a);
};
goog.setCssNameMapping = function(a, b) {
  goog.cssNameMapping_ = a;
  goog.cssNameMappingStyle_ = b;
};
!COMPILED && goog.global.CLOSURE_CSS_NAME_MAPPING && (goog.cssNameMapping_ = goog.global.CLOSURE_CSS_NAME_MAPPING);
goog.getMsg = function(a, b) {
  var c = b || {}, d;
  for (d in c) {
    var e = ("" + c[d]).replace(/\$/g, "$$$$");
    a = a.replace(new RegExp("\\{\\$" + d + "\\}", "gi"), e);
  }
  return a;
};
goog.getMsgWithFallback = function(a, b) {
  return a;
};
goog.exportSymbol = function(a, b, c) {
  goog.exportPath_(a, b, c);
};
goog.exportProperty = function(a, b, c) {
  a[b] = c;
};
goog.inherits = function(a, b) {
  function c() {
  }
  c.prototype = b.prototype;
  a.superClass_ = b.prototype;
  a.prototype = new c;
  a.prototype.constructor = a;
  a.base = function(a, c, f) {
    var g = Array.prototype.slice.call(arguments, 2);
    return b.prototype[c].apply(a, g);
  };
};
goog.base = function(a, b, c) {
  var d = arguments.callee.caller;
  if (goog.STRICT_MODE_COMPATIBLE || goog.DEBUG && !d) {
    throw Error("arguments.caller not defined.  goog.base() cannot be used with strict mode code. See http://www.ecma-international.org/ecma-262/5.1/#sec-C");
  }
  if (d.superClass_) {
    return d.superClass_.constructor.apply(a, Array.prototype.slice.call(arguments, 1));
  }
  for (var e = Array.prototype.slice.call(arguments, 2), f = !1, g = a.constructor;g;g = g.superClass_ && g.superClass_.constructor) {
    if (g.prototype[b] === d) {
      f = !0;
    } else {
      if (f) {
        return g.prototype[b].apply(a, e);
      }
    }
  }
  if (a[b] === d) {
    return a.constructor.prototype[b].apply(a, e);
  }
  throw Error("goog.base called from a method of one name to a method of a different name");
};
goog.scope = function(a) {
  a.call(goog.global);
};
var axs = {};
axs.browserUtils = {};
axs.browserUtils.matchSelector = function(a, b) {
  return a.webkitMatchesSelector ? a.webkitMatchesSelector(b) : a.mozMatchesSelector ? a.mozMatchesSelector(b) : !1;
};
axs.constants = {};
axs.constants.ARIA_ROLES = {alert:{namefrom:["author"], parent:["region"]}, alertdialog:{namefrom:["author"], namerequired:!0, parent:["alert", "dialog"]}, application:{namefrom:["author"], namerequired:!0, parent:["landmark"]}, article:{namefrom:["author"], parent:["document", "region"]}, banner:{namefrom:["author"], parent:["landmark"]}, button:{childpresentational:!0, namefrom:["contents", "author"], namerequired:!0, parent:["command"], properties:["aria-expanded", "aria-pressed"]}, checkbox:{namefrom:["contents", 
"author"], namerequired:!0, parent:["input"], requiredProperties:["aria-checked"], properties:["aria-checked"]}, columnheader:{namefrom:["contents", "author"], namerequired:!0, parent:["gridcell", "sectionhead", "widget"], properties:["aria-sort"]}, combobox:{mustcontain:["listbox", "textbox"], namefrom:["author"], namerequired:!0, parent:["select"], requiredProperties:["aria-expanded"], properties:["aria-expanded", "aria-autocomplete", "aria-required"]}, command:{"abstract":!0, namefrom:["author"], 
parent:["widget"]}, complementary:{namefrom:["author"], parent:["landmark"]}, composite:{"abstract":!0, childpresentational:!1, namefrom:["author"], parent:["widget"], properties:["aria-activedescendant"]}, contentinfo:{namefrom:["author"], parent:["landmark"]}, definition:{namefrom:["author"], parent:["section"]}, dialog:{namefrom:["author"], namerequired:!0, parent:["window"]}, directory:{namefrom:["contents", "author"], parent:["list"]}, document:{namefrom:[" author"], namerequired:!0, parent:["structure"], 
properties:["aria-expanded"]}, form:{namefrom:["author"], parent:["landmark"]}, grid:{mustcontain:["row", "rowgroup"], namefrom:["author"], namerequired:!0, parent:["composite", "region"], properties:["aria-level", "aria-multiselectable", "aria-readonly"]}, gridcell:{namefrom:["contents", "author"], namerequired:!0, parent:["section", "widget"], properties:["aria-readonly", "aria-required", "aria-selected"]}, group:{namefrom:[" author"], parent:["section"], properties:["aria-activedescendant"]}, 
heading:{namerequired:!0, parent:["sectionhead"], properties:["aria-level"]}, img:{childpresentational:!0, namefrom:["author"], namerequired:!0, parent:["section"]}, input:{"abstract":!0, namefrom:["author"], parent:["widget"]}, landmark:{"abstract":!0, namefrom:["contents", "author"], namerequired:!1, parent:["region"]}, link:{namefrom:["contents", "author"], namerequired:!0, parent:["command"], properties:["aria-expanded"]}, list:{mustcontain:["group", "listitem"], namefrom:["author"], parent:["region"]}, 
listbox:{mustcontain:["option"], namefrom:["author"], namerequired:!0, parent:["list", "select"], properties:["aria-multiselectable", "aria-required"]}, listitem:{namefrom:["contents", "author"], namerequired:!0, parent:["section"], properties:["aria-level", "aria-posinset", "aria-setsize"]}, log:{namefrom:[" author"], namerequired:!0, parent:["region"]}, main:{namefrom:["author"], parent:["landmark"]}, marquee:{namerequired:!0, parent:["section"]}, math:{childpresentational:!0, namefrom:["author"], 
parent:["section"]}, menu:{mustcontain:["group", "menuitemradio", "menuitem", "menuitemcheckbox"], namefrom:["author"], namerequired:!0, parent:["list", "select"]}, menubar:{namefrom:["author"], parent:["menu"]}, menuitem:{namefrom:["contents", "author"], namerequired:!0, parent:["command"]}, menuitemcheckbox:{namefrom:["contents", "author"], namerequired:!0, parent:["checkbox", "menuitem"]}, menuitemradio:{namefrom:["contents", "author"], namerequired:!0, parent:["menuitemcheckbox", "radio"]}, navigation:{namefrom:["author"], 
parent:["landmark"]}, note:{namefrom:["author"], parent:["section"]}, option:{namefrom:["contents", "author"], namerequired:!0, parent:["input"], properties:["aria-checked", "aria-posinset", "aria-selected", "aria-setsize"]}, presentation:{parent:["structure"]}, progressbar:{childpresentational:!0, namefrom:["author"], namerequired:!0, parent:["range"]}, radio:{namefrom:["contents", "author"], namerequired:!0, parent:["checkbox", "option"]}, radiogroup:{mustcontain:["radio"], namefrom:["author"], 
namerequired:!0, parent:["select"], properties:["aria-required"]}, range:{"abstract":!0, namefrom:["author"], parent:["widget"], properties:["aria-valuemax", "aria-valuemin", "aria-valuenow", "aria-valuetext"]}, region:{namefrom:[" author"], parent:["section"]}, roletype:{"abstract":!0, properties:"aria-atomic aria-busy aria-controls aria-describedby aria-disabled aria-dropeffect aria-flowto aria-grabbed aria-haspopup aria-hidden aria-invalid aria-label aria-labelledby aria-live aria-owns aria-relevant".split(" ")}, 
row:{mustcontain:["columnheader", "gridcell", "rowheader"], namefrom:["contents", "author"], parent:["group", "widget"], properties:["aria-level", "aria-selected"]}, rowgroup:{mustcontain:["row"], namefrom:["contents", "author"], parent:["group"]}, rowheader:{namefrom:["contents", "author"], namerequired:!0, parent:["gridcell", "sectionhead", "widget"], properties:["aria-sort"]}, search:{namefrom:["author"], parent:["landmark"]}, section:{"abstract":!0, namefrom:["contents", "author"], parent:["structure"], 
properties:["aria-expanded"]}, sectionhead:{"abstract":!0, namefrom:["contents", "author"], parent:["structure"], properties:["aria-expanded"]}, select:{"abstract":!0, namefrom:["author"], parent:["composite", "group", "input"]}, separator:{childpresentational:!0, namefrom:["author"], parent:["structure"], properties:["aria-expanded", "aria-orientation"]}, scrollbar:{childpresentational:!0, namefrom:["author"], namerequired:!1, parent:["input", "range"], requiredProperties:["aria-controls", "aria-orientation", 
"aria-valuemax", "aria-valuemin", "aria-valuenow"], properties:["aria-controls", "aria-orientation", "aria-valuemax", "aria-valuemin", "aria-valuenow"]}, slider:{childpresentational:!0, namefrom:["author"], namerequired:!0, parent:["input", "range"], requiredProperties:["aria-valuemax", "aria-valuemin", "aria-valuenow"], properties:["aria-valuemax", "aria-valuemin", "aria-valuenow", "aria-orientation"]}, spinbutton:{namefrom:["author"], namerequired:!0, parent:["input", "range"], requiredProperties:["aria-valuemax", 
"aria-valuemin", "aria-valuenow"], properties:["aria-valuemax", "aria-valuemin", "aria-valuenow", "aria-required"]}, status:{parent:["region"]}, structure:{"abstract":!0, parent:["roletype"]}, tab:{namefrom:["contents", "author"], parent:["sectionhead", "widget"], properties:["aria-selected"]}, tablist:{mustcontain:["tab"], namefrom:["author"], parent:["composite", "directory"], properties:["aria-level"]}, tabpanel:{namefrom:["author"], namerequired:!0, parent:["region"]}, textbox:{namefrom:["author"], 
namerequired:!0, parent:["input"], properties:["aria-activedescendant", "aria-autocomplete", "aria-multiline", "aria-readonly", "aria-required"]}, timer:{namefrom:["author"], namerequired:!0, parent:["status"]}, toolbar:{namefrom:["author"], parent:["group"]}, tooltip:{namerequired:!0, parent:["section"]}, tree:{mustcontain:["group", "treeitem"], namefrom:["author"], namerequired:!0, parent:["select"], properties:["aria-multiselectable", "aria-required"]}, treegrid:{mustcontain:["row"], namefrom:["author"], 
namerequired:!0, parent:["grid", "tree"]}, treeitem:{namefrom:["contents", "author"], namerequired:!0, parent:["listitem", "option"]}, widget:{"abstract":!0, parent:["roletype"]}, window:{"abstract":!0, namefrom:[" author"], parent:["roletype"], properties:["aria-expanded"]}};
axs.constants.WIDGET_ROLES = {};
axs.constants.addAllParentRolesToSet_ = function(a, b) {
  if (a.parent) {
    for (var c = a.parent, d = 0;d < c.length;d++) {
      var e = c[d];
      b[e] = !0;
      axs.constants.addAllParentRolesToSet_(axs.constants.ARIA_ROLES[e], b);
    }
  }
};
axs.constants.addAllPropertiesToSet_ = function(a, b, c) {
  var d = a[b];
  if (d) {
    for (var e = 0;e < d.length;e++) {
      c[d[e]] = !0;
    }
  }
  if (a.parent) {
    for (a = a.parent, d = 0;d < a.length;d++) {
      axs.constants.addAllPropertiesToSet_(axs.constants.ARIA_ROLES[a[d]], b, c);
    }
  }
};
for (var roleName in axs.constants.ARIA_ROLES) {
  var role = axs.constants.ARIA_ROLES[roleName], propertiesSet = {};
  axs.constants.addAllPropertiesToSet_(role, "properties", propertiesSet);
  role.propertiesSet = propertiesSet;
  var requiredPropertiesSet = {};
  axs.constants.addAllPropertiesToSet_(role, "requiredProperties", requiredPropertiesSet);
  role.requiredPropertiesSet = requiredPropertiesSet;
  var parentRolesSet = {};
  axs.constants.addAllParentRolesToSet_(role, parentRolesSet);
  role.allParentRolesSet = parentRolesSet;
  "widget" in parentRolesSet && (axs.constants.WIDGET_ROLES[roleName] = role);
}
axs.constants.ARIA_PROPERTIES = {activedescendant:{type:"property", valueType:"idref"}, atomic:{defaultValue:"false", type:"property", valueType:"boolean"}, autocomplete:{defaultValue:"none", type:"property", valueType:"token", values:["inline", "list", "both", "none"]}, busy:{defaultValue:"false", type:"state", valueType:"boolean"}, checked:{defaultValue:"undefined", type:"state", valueType:"token", values:["true", "false", "mixed", "undefined"]}, controls:{type:"property", valueType:"idref_list"}, 
describedby:{type:"property", valueType:"idref_list"}, disabled:{defaultValue:"false", type:"state", valueType:"boolean"}, dropeffect:{defaultValue:"none", type:"property", valueType:"token_list", values:"copy move link execute popup none".split(" ")}, expanded:{defaultValue:"undefined", type:"state", valueType:"token", values:["true", "false", "undefined"]}, flowto:{type:"property", valueType:"idref_list"}, grabbed:{defaultValue:"undefined", type:"state", valueType:"token", values:["true", "false", 
"undefined"]}, haspopup:{defaultValue:"false", type:"property", valueType:"boolean"}, hidden:{defaultValue:"false", type:"state", valueType:"boolean"}, invalid:{defaultValue:"false", type:"state", valueType:"token", values:["grammar", "false", "spelling", "true"]}, label:{type:"property", valueType:"string"}, labelledby:{type:"property", valueType:"idref_list"}, level:{type:"property", valueType:"integer"}, live:{defaultValue:"off", type:"property", valueType:"token", values:["off", "polite", "assertive"]}, 
multiline:{defaultValue:"false", type:"property", valueType:"boolean"}, multiselectable:{defaultValue:"false", type:"property", valueType:"boolean"}, orientation:{defaultValue:"vertical", type:"property", valueType:"token", values:["horizontal", "vertical"]}, owns:{type:"property", valueType:"idref_list"}, posinset:{type:"property", valueType:"integer"}, pressed:{defaultValue:"undefined", type:"state", valueType:"token", values:["true", "false", "mixed", "undefined"]}, readonly:{defaultValue:"false", 
type:"property", valueType:"boolean"}, relevant:{defaultValue:"additions text", type:"property", valueType:"token_list", values:["additions", "removals", "text", "all"]}, required:{defaultValue:"false", type:"property", valueType:"boolean"}, selected:{defaultValue:"undefined", type:"state", valueType:"token", values:["true", "false", "undefined"]}, setsize:{type:"property", valueType:"integer"}, sort:{defaultValue:"none", type:"property", valueType:"token", values:["ascending", "descending", "none", 
"other"]}, valuemax:{type:"property", valueType:"decimal"}, valuemin:{type:"property", valueType:"decimal"}, valuenow:{type:"property", valueType:"decimal"}, valuetext:{type:"property", valueType:"string"}};
axs.constants.GLOBAL_PROPERTIES = "aria-atomic aria-busy aria-controls aria-describedby aria-disabled aria-dropeffect aria-flowto aria-grabbed aria-haspopup aria-hidden aria-invalid aria-label aria-labelledby aria-live aria-owns aria-relevant".split(" ");
axs.constants.NO_ROLE_NAME = " ";
axs.constants.WIDGET_ROLE_TO_NAME = {alert:"aria_role_alert", alertdialog:"aria_role_alertdialog", button:"aria_role_button", checkbox:"aria_role_checkbox", columnheader:"aria_role_columnheader", combobox:"aria_role_combobox", dialog:"aria_role_dialog", grid:"aria_role_grid", gridcell:"aria_role_gridcell", link:"aria_role_link", listbox:"aria_role_listbox", log:"aria_role_log", marquee:"aria_role_marquee", menu:"aria_role_menu", menubar:"aria_role_menubar", menuitem:"aria_role_menuitem", menuitemcheckbox:"aria_role_menuitemcheckbox", 
menuitemradio:"aria_role_menuitemradio", option:axs.constants.NO_ROLE_NAME, progressbar:"aria_role_progressbar", radio:"aria_role_radio", radiogroup:"aria_role_radiogroup", rowheader:"aria_role_rowheader", scrollbar:"aria_role_scrollbar", slider:"aria_role_slider", spinbutton:"aria_role_spinbutton", status:"aria_role_status", tab:"aria_role_tab", tabpanel:"aria_role_tabpanel", textbox:"aria_role_textbox", timer:"aria_role_timer", toolbar:"aria_role_toolbar", tooltip:"aria_role_tooltip", treeitem:"aria_role_treeitem"};
axs.constants.STRUCTURE_ROLE_TO_NAME = {article:"aria_role_article", application:"aria_role_application", banner:"aria_role_banner", columnheader:"aria_role_columnheader", complementary:"aria_role_complementary", contentinfo:"aria_role_contentinfo", definition:"aria_role_definition", directory:"aria_role_directory", document:"aria_role_document", form:"aria_role_form", group:"aria_role_group", heading:"aria_role_heading", img:"aria_role_img", list:"aria_role_list", listitem:"aria_role_listitem", 
main:"aria_role_main", math:"aria_role_math", navigation:"aria_role_navigation", note:"aria_role_note", region:"aria_role_region", rowheader:"aria_role_rowheader", search:"aria_role_search", separator:"aria_role_separator"};
axs.constants.ATTRIBUTE_VALUE_TO_STATUS = [{name:"aria-autocomplete", values:{inline:"aria_autocomplete_inline", list:"aria_autocomplete_list", both:"aria_autocomplete_both"}}, {name:"aria-checked", values:{"true":"aria_checked_true", "false":"aria_checked_false", mixed:"aria_checked_mixed"}}, {name:"aria-disabled", values:{"true":"aria_disabled_true"}}, {name:"aria-expanded", values:{"true":"aria_expanded_true", "false":"aria_expanded_false"}}, {name:"aria-invalid", values:{"true":"aria_invalid_true", 
grammar:"aria_invalid_grammar", spelling:"aria_invalid_spelling"}}, {name:"aria-multiline", values:{"true":"aria_multiline_true"}}, {name:"aria-multiselectable", values:{"true":"aria_multiselectable_true"}}, {name:"aria-pressed", values:{"true":"aria_pressed_true", "false":"aria_pressed_false", mixed:"aria_pressed_mixed"}}, {name:"aria-readonly", values:{"true":"aria_readonly_true"}}, {name:"aria-required", values:{"true":"aria_required_true"}}, {name:"aria-selected", values:{"true":"aria_selected_true", 
"false":"aria_selected_false"}}];
axs.constants.INPUT_TYPE_TO_INFORMATION_TABLE_MSG = {button:"input_type_button", checkbox:"input_type_checkbox", color:"input_type_color", datetime:"input_type_datetime", "datetime-local":"input_type_datetime_local", date:"input_type_date", email:"input_type_email", file:"input_type_file", image:"input_type_image", month:"input_type_month", number:"input_type_number", password:"input_type_password", radio:"input_type_radio", range:"input_type_range", reset:"input_type_reset", search:"input_type_search", 
submit:"input_type_submit", tel:"input_type_tel", text:"input_type_text", url:"input_type_url", week:"input_type_week"};
axs.constants.TAG_TO_INFORMATION_TABLE_VERBOSE_MSG = {A:"tag_link", BUTTON:"tag_button", H1:"tag_h1", H2:"tag_h2", H3:"tag_h3", H4:"tag_h4", H5:"tag_h5", H6:"tag_h6", LI:"tag_li", OL:"tag_ol", SELECT:"tag_select", TEXTAREA:"tag_textarea", UL:"tag_ul", SECTION:"tag_section", NAV:"tag_nav", ARTICLE:"tag_article", ASIDE:"tag_aside", HGROUP:"tag_hgroup", HEADER:"tag_header", FOOTER:"tag_footer", TIME:"tag_time", MARK:"tag_mark"};
axs.constants.TAG_TO_INFORMATION_TABLE_BRIEF_MSG = {BUTTON:"tag_button", SELECT:"tag_select", TEXTAREA:"tag_textarea"};
axs.constants.MIXED_VALUES = {"true":!0, "false":!0, mixed:!0};
(function() {
  for (var a in axs.constants.ARIA_PROPERTIES) {
    var b = axs.constants.ARIA_PROPERTIES[a];
    if (b.values) {
      for (var c = {}, d = 0;d < b.values.length;d++) {
        c[b.values[d]] = !0;
      }
      b.valuesSet = c;
    }
  }
})();
axs.constants.Severity = {INFO:"Info", WARNING:"Warning", SEVERE:"Severe"};
axs.constants.AuditResult = {PASS:"PASS", FAIL:"FAIL", NA:"NA"};
axs.constants.InlineElements = {TT:!0, I:!0, B:!0, BIG:!0, SMALL:!0, EM:!0, STRONG:!0, DFN:!0, CODE:!0, SAMP:!0, KBD:!0, VAR:!0, CITE:!0, ABBR:!0, ACRONYM:!0, A:!0, IMG:!0, OBJECT:!0, BR:!0, SCRIPT:!0, MAP:!0, Q:!0, SUB:!0, SUP:!0, SPAN:!0, BDO:!0, INPUT:!0, SELECT:!0, TEXTAREA:!0, LABEL:!0, BUTTON:!0};
axs.utils = {};
axs.utils.FOCUSABLE_ELEMENTS_SELECTOR = "input:not([type=hidden]):not([disabled]),select:not([disabled]),textarea:not([disabled]),button:not([disabled]),a[href],iframe,[tabindex]";
axs.utils.Color = function(a, b, c, d) {
  this.red = a;
  this.green = b;
  this.blue = c;
  this.alpha = d;
};
axs.utils.calculateContrastRatio = function(a, b) {
  if (!a || !b) {
    return null;
  }
  1 > a.alpha && (a = axs.utils.flattenColors(a, b));
  var c = axs.utils.calculateLuminance(a), d = axs.utils.calculateLuminance(b);
  return(Math.max(c, d) + .05) / (Math.min(c, d) + .05);
};
axs.utils.luminanceRatio = function(a, b) {
  return(Math.max(a, b) + .05) / (Math.min(a, b) + .05);
};
axs.utils.parentElement = function(a) {
  if (!a) {
    return null;
  }
  if (a.nodeType == Node.DOCUMENT_FRAGMENT_NODE) {
    return a.host;
  }
  var b = a.parentElement;
  if (b) {
    return b;
  }
  a = a.parentNode;
  if (!a) {
    return null;
  }
  switch(a.nodeType) {
    case Node.ELEMENT_NODE:
      return a;
    case Node.DOCUMENT_FRAGMENT_NODE:
      return a.host;
    default:
      return null;
  }
};
axs.utils.asElement = function(a) {
  switch(a.nodeType) {
    case Node.COMMENT_NODE:
      return null;
    case Node.ELEMENT_NODE:
      if ("script" == a.tagName.toLowerCase()) {
        return null;
      }
      break;
    case Node.TEXT_NODE:
      a = axs.utils.parentElement(a);
      break;
    default:
      return console.warn("Unhandled node type: ", a.nodeType), null;
  }
  return a;
};
axs.utils.elementIsTransparent = function(a) {
  return "0" == a.style.opacity;
};
axs.utils.elementHasZeroArea = function(a) {
  a = a.getBoundingClientRect();
  var b = a.top - a.bottom;
  return a.right - a.left && b ? !1 : !0;
};
axs.utils.elementIsOutsideScrollArea = function(a) {
  for (var b = axs.utils.parentElement(a), c = a.ownerDocument.defaultView;b != c.document.body;) {
    if (axs.utils.isClippedBy(a, b)) {
      return!0;
    }
    if (axs.utils.canScrollTo(a, b) && !axs.utils.elementIsOutsideScrollArea(b)) {
      return!1;
    }
    b = axs.utils.parentElement(b);
  }
  return!axs.utils.canScrollTo(a, c.document.body);
};
axs.utils.canScrollTo = function(a, b) {
  var c = a.getBoundingClientRect(), d = b.getBoundingClientRect(), e = d.top, f = d.left, g = e - b.scrollTop, e = e - b.scrollTop + b.scrollHeight, h = f - b.scrollLeft + b.scrollWidth;
  if (c.right < f - b.scrollLeft || c.bottom < g || c.left > h || c.top > e) {
    return!1;
  }
  f = a.ownerDocument.defaultView;
  g = f.getComputedStyle(b);
  return c.left > d.right || c.top > d.bottom ? "scroll" == g.overflow || "auto" == g.overflow || b instanceof f.HTMLBodyElement : !0;
};
axs.utils.isClippedBy = function(a, b) {
  var c = a.getBoundingClientRect(), d = b.getBoundingClientRect(), e = d.top - b.scrollTop, f = d.left - b.scrollLeft, g = a.ownerDocument.defaultView.getComputedStyle(b);
  return(c.right < d.left || c.bottom < d.top || c.left > d.right || c.top > d.bottom) && "hidden" == g.overflow ? !0 : c.right < f || c.bottom < e ? "visible" != g.overflow : !1;
};
axs.utils.isAncestor = function(a, b) {
  return null == b ? !1 : b === a ? !0 : axs.utils.isAncestor(a, b.parentNode);
};
axs.utils.overlappingElements = function(a) {
  if (axs.utils.elementHasZeroArea(a)) {
    return null;
  }
  for (var b = [], c = a.getClientRects(), d = 0;d < c.length;d++) {
    var e = c[d], e = document.elementFromPoint((e.left + e.right) / 2, (e.top + e.bottom) / 2);
    if (null != e && e != a && !axs.utils.isAncestor(e, a) && !axs.utils.isAncestor(a, e)) {
      var f = window.getComputedStyle(e, null);
      f && (f = axs.utils.getBgColor(f, e)) && 0 < f.alpha && 0 > b.indexOf(e) && b.push(e);
    }
  }
  return b;
};
axs.utils.elementIsHtmlControl = function(a) {
  var b = a.ownerDocument.defaultView;
  return a instanceof b.HTMLButtonElement || a instanceof b.HTMLInputElement || a instanceof b.HTMLSelectElement || a instanceof b.HTMLTextAreaElement ? !0 : !1;
};
axs.utils.elementIsAriaWidget = function(a) {
  return a.hasAttribute("role") && (a = a.getAttribute("role")) && (a = axs.constants.ARIA_ROLES[a]) && "widget" in a.allParentRolesSet ? !0 : !1;
};
axs.utils.elementIsVisible = function(a) {
  return axs.utils.elementIsTransparent(a) || axs.utils.elementHasZeroArea(a) || axs.utils.elementIsOutsideScrollArea(a) || axs.utils.overlappingElements(a).length ? !1 : !0;
};
axs.utils.isLargeFont = function(a) {
  var b = a.fontSize;
  a = "bold" == a.fontWeight;
  var c = b.match(/(\d+)px/);
  if (c) {
    b = parseInt(c[1], 10);
    if (c = window.getComputedStyle(document.body, null).fontSize.match(/(\d+)px/)) {
      var d = parseInt(c[1], 10), c = 1.2 * d, d = 1.5 * d
    } else {
      c = 19.2, d = 24;
    }
    return a && b >= c || b >= d;
  }
  if (c = b.match(/(\d+)em/)) {
    return b = parseInt(c[1], 10), a && 1.2 <= b || 1.5 <= b ? !0 : !1;
  }
  if (c = b.match(/(\d+)%/)) {
    return b = parseInt(c[1], 10), a && 120 <= b || 150 <= b ? !0 : !1;
  }
  if (c = b.match(/(\d+)pt/)) {
    if (b = parseInt(c[1], 10), a && 14 <= b || 18 <= b) {
      return!0;
    }
  }
  return!1;
};
axs.utils.getBgColor = function(a, b) {
  var c = axs.utils.parseColor(a.backgroundColor);
  if (!c) {
    return null;
  }
  1 > a.opacity && (c.alpha *= a.opacity);
  if (1 > c.alpha) {
    var d = axs.utils.getParentBgColor(b);
    if (null == d) {
      return null;
    }
    c = axs.utils.flattenColors(c, d);
  }
  return c;
};
axs.utils.getParentBgColor = function(a) {
  var b = a;
  a = [];
  for (var c = null;b = axs.utils.parentElement(b);) {
    var d = window.getComputedStyle(b, null);
    if (d) {
      var e = axs.utils.parseColor(d.backgroundColor);
      if (e && (1 > d.opacity && (e.alpha *= d.opacity), 0 != e.alpha && (a.push(e), 1 == e.alpha))) {
        c = !0;
        break;
      }
    }
  }
  c || a.push(new axs.utils.Color(255, 255, 255, 1));
  for (b = a.pop();a.length;) {
    c = a.pop(), b = axs.utils.flattenColors(c, b);
  }
  return b;
};
axs.utils.getFgColor = function(a, b, c) {
  var d = axs.utils.parseColor(a.color);
  if (!d) {
    return null;
  }
  1 > d.alpha && (d = axs.utils.flattenColors(d, c));
  1 > a.opacity && (b = axs.utils.getParentBgColor(b), d.alpha *= a.opacity, d = axs.utils.flattenColors(d, b));
  return d;
};
axs.utils.parseColor = function(a) {
  var b = a.match(/^rgb\((\d+), (\d+), (\d+)\)$/);
  if (b) {
    a = parseInt(b[1], 10);
    var c = parseInt(b[2], 10), b = parseInt(b[3], 10), d;
    return new axs.utils.Color(a, c, b, 1);
  }
  return(b = a.match(/^rgba\((\d+), (\d+), (\d+), (\d+(\.\d+)?)\)/)) ? (d = parseInt(b[4], 10), a = parseInt(b[1], 10), c = parseInt(b[2], 10), b = parseInt(b[3], 10), new axs.utils.Color(a, c, b, d)) : null;
};
axs.utils.colorChannelToString = function(a) {
  a = Math.round(a);
  return 15 >= a ? "0" + a.toString(16) : a.toString(16);
};
axs.utils.colorToString = function(a) {
  return 1 == a.alpha ? "#" + axs.utils.colorChannelToString(a.red) + axs.utils.colorChannelToString(a.green) + axs.utils.colorChannelToString(a.blue) : "rgba(" + [a.red, a.green, a.blue, a.alpha].join() + ")";
};
axs.utils.luminanceFromContrastRatio = function(a, b, c) {
  return c ? (a + .05) * b - .05 : (a + .05) / b - .05;
};
axs.utils.translateColor = function(a, b) {
  var c = a[0], c = (b - c) / ((c > b ? 0 : 1) - c);
  return axs.utils.fromYCC([b, a[1] - a[1] * c, a[2] - a[2] * c]);
};
axs.utils.suggestColors = function(a, b, c, d) {
  if (!axs.utils.isLowContrast(c, d, !0)) {
    return null;
  }
  var e = {}, f = axs.utils.calculateLuminance(a), g = axs.utils.calculateLuminance(b), h = axs.utils.isLargeFont(d) ? 3 : 4.5, k = axs.utils.isLargeFont(d) ? 4.5 : 7, m = g > f, l = axs.utils.luminanceFromContrastRatio(f, h + .02, m), n = axs.utils.luminanceFromContrastRatio(f, k + .02, m), q = axs.utils.toYCC(b);
  if (axs.utils.isLowContrast(c, d, !1) && 1 >= l && 0 <= l) {
    var p = axs.utils.translateColor(q, l), l = axs.utils.calculateContrastRatio(p, a);
    axs.utils.calculateLuminance(p);
    f = {};
    f.fg = axs.utils.colorToString(p);
    f.bg = axs.utils.colorToString(a);
    f.contrast = l.toFixed(2);
    e.AA = f;
  }
  axs.utils.isLowContrast(c, d, !0) && 1 >= n && 0 <= n && (n = axs.utils.translateColor(q, n), l = axs.utils.calculateContrastRatio(n, a), f = {}, f.fg = axs.utils.colorToString(n), f.bg = axs.utils.colorToString(a), f.contrast = l.toFixed(2), e.AAA = f);
  h = axs.utils.luminanceFromContrastRatio(g, h + .02, !m);
  g = axs.utils.luminanceFromContrastRatio(g, k + .02, !m);
  a = axs.utils.toYCC(a);
  !("AA" in e) && axs.utils.isLowContrast(c, d, !1) && 1 >= h && 0 <= h && (k = axs.utils.translateColor(a, h), l = axs.utils.calculateContrastRatio(b, k), f = {}, f.bg = axs.utils.colorToString(k), f.fg = axs.utils.colorToString(b), f.contrast = l.toFixed(2), e.AA = f);
  !("AAA" in e) && axs.utils.isLowContrast(c, d, !0) && 1 >= g && 0 <= g && (c = axs.utils.translateColor(a, g), l = axs.utils.calculateContrastRatio(b, c), f = {}, f.bg = axs.utils.colorToString(c), f.fg = axs.utils.colorToString(b), f.contrast = l.toFixed(2), e.AAA = f);
  return e;
};
axs.utils.flattenColors = function(a, b) {
  var c = a.alpha;
  return new axs.utils.Color((1 - c) * b.red + c * a.red, (1 - c) * b.green + c * a.green, (1 - c) * b.blue + c * a.blue, a.alpha + b.alpha * (1 - a.alpha));
};
axs.utils.calculateLuminance = function(a) {
  return axs.utils.toYCC(a)[0];
};
axs.utils.RGBToYCCMatrix = function(a, b) {
  return[[a, 1 - a - b, b], [-a / (2 - 2 * b), (a + b - 1) / (2 - 2 * b), (1 - b) / (2 - 2 * b)], [(1 - a) / (2 - 2 * a), (a + b - 1) / (2 - 2 * a), -b / (2 - 2 * a)]];
};
axs.utils.invert3x3Matrix = function(a) {
  var b = a[0][0], c = a[0][1], d = a[0][2], e = a[1][0], f = a[1][1], g = a[1][2], h = a[2][0], k = a[2][1];
  a = a[2][2];
  return axs.utils.scalarMultiplyMatrix([[f * a - g * k, d * k - c * a, c * g - d * f], [g * h - e * a, b * a - d * h, d * e - b * g], [e * k - f * h, h * c - b * k, b * f - c * e]], 1 / (b * (f * a - g * k) - c * (a * e - g * h) + d * (e * k - f * h)));
};
axs.utils.scalarMultiplyMatrix = function(a, b) {
  for (var c = [[], [], []], d = 0;3 > d;d++) {
    for (var e = 0;3 > e;e++) {
      c[d][e] = a[d][e] * b;
    }
  }
  return c;
};
axs.utils.kR = .2126;
axs.utils.kB = .0722;
axs.utils.YCC_MATRIX = axs.utils.RGBToYCCMatrix(axs.utils.kR, axs.utils.kB);
axs.utils.INVERTED_YCC_MATRIX = axs.utils.invert3x3Matrix(axs.utils.YCC_MATRIX);
axs.utils.convertColor = function(a, b) {
  var c = b[0], d = b[1], e = b[2];
  return[a[0][0] * c + a[0][1] * d + a[0][2] * e, a[1][0] * c + a[1][1] * d + a[1][2] * e, a[2][0] * c + a[2][1] * d + a[2][2] * e];
};
axs.utils.multiplyMatrices = function(a, b) {
  for (var c = [[], [], []], d = 0;3 > d;d++) {
    for (var e = 0;3 > e;e++) {
      c[d][e] = a[d][0] * b[0][e] + a[d][1] * b[1][e] + a[d][2] * b[2][e];
    }
  }
  return c;
};
axs.utils.toYCC = function(a) {
  var b = a.red / 255, c = a.green / 255;
  a = a.blue / 255;
  b = .03928 >= b ? b / 12.92 : Math.pow((b + .055) / 1.055, 2.4);
  c = .03928 >= c ? c / 12.92 : Math.pow((c + .055) / 1.055, 2.4);
  a = .03928 >= a ? a / 12.92 : Math.pow((a + .055) / 1.055, 2.4);
  return axs.utils.convertColor(axs.utils.YCC_MATRIX, [b, c, a]);
};
axs.utils.fromYCC = function(a) {
  var b = axs.utils.convertColor(axs.utils.INVERTED_YCC_MATRIX, a), c = b[0];
  a = b[1];
  b = b[2];
  c = .00303949 >= c ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - .055;
  a = .00303949 >= a ? 12.92 * a : 1.055 * Math.pow(a, 1 / 2.4) - .055;
  b = .00303949 >= b ? 12.92 * b : 1.055 * Math.pow(b, 1 / 2.4) - .055;
  c = Math.min(Math.max(Math.round(255 * c), 0), 255);
  a = Math.min(Math.max(Math.round(255 * a), 0), 255);
  b = Math.min(Math.max(Math.round(255 * b), 0), 255);
  return new axs.utils.Color(c, a, b, 1);
};
axs.utils.scalarMultiplyMatrix = function(a, b) {
  for (var c = [[], [], []], d = 0;3 > d;d++) {
    for (var e = 0;3 > e;e++) {
      c[d][e] = a[d][e] * b;
    }
  }
  return c;
};
axs.utils.multiplyMatrices = function(a, b) {
  for (var c = [[], [], []], d = 0;3 > d;d++) {
    for (var e = 0;3 > e;e++) {
      c[d][e] = a[d][0] * b[0][e] + a[d][1] * b[1][e] + a[d][2] * b[2][e];
    }
  }
  return c;
};
axs.utils.getContrastRatioForElement = function(a) {
  var b = window.getComputedStyle(a, null);
  return axs.utils.getContrastRatioForElementWithComputedStyle(b, a);
};
axs.utils.getContrastRatioForElementWithComputedStyle = function(a, b) {
  if (axs.utils.isElementHidden(b)) {
    return null;
  }
  var c = axs.utils.getBgColor(a, b);
  if (!c) {
    return null;
  }
  var d = axs.utils.getFgColor(a, b, c);
  return d ? axs.utils.calculateContrastRatio(d, c) : null;
};
axs.utils.isNativeTextElement = function(a) {
  var b = a.tagName.toLowerCase();
  a = a.type ? a.type.toLowerCase() : "";
  if ("textarea" == b) {
    return!0;
  }
  if ("input" != b) {
    return!1;
  }
  switch(a) {
    case "email":
    ;
    case "number":
    ;
    case "password":
    ;
    case "search":
    ;
    case "text":
    ;
    case "tel":
    ;
    case "url":
    ;
    case "":
      return!0;
    default:
      return!1;
  }
};
axs.utils.isLowContrast = function(a, b, c) {
  a = Math.round(10 * a) / 10;
  return c ? 4.5 > a || !axs.utils.isLargeFont(b) && 7 > a : 3 > a || !axs.utils.isLargeFont(b) && 4.5 > a;
};
axs.utils.hasLabel = function(a) {
  var b = a.tagName.toLowerCase(), c = a.type ? a.type.toLowerCase() : "";
  if (a.hasAttribute("aria-label") || a.hasAttribute("title") || "img" == b && a.hasAttribute("alt") || "input" == b && "image" == c && a.hasAttribute("alt") || "input" == b && ("submit" == c || "reset" == c) || a.hasAttribute("aria-labelledby") || axs.utils.isNativeTextElement(a) && a.hasAttribute("placeholder") || a.hasAttribute("id") && 0 < document.querySelectorAll('label[for="' + a.id + '"]').length) {
    return!0;
  }
  for (b = axs.utils.parentElement(a);b;) {
    if ("label" == b.tagName.toLowerCase() && b.control == a) {
      return!0;
    }
    b = axs.utils.parentElement(b);
  }
  return!1;
};
axs.utils.isElementHidden = function(a) {
  if (!(a instanceof a.ownerDocument.defaultView.HTMLElement)) {
    return!1;
  }
  if (a.hasAttribute("chromevoxignoreariahidden")) {
    var b = !0
  }
  var c = window.getComputedStyle(a, null);
  return "none" == c.display || "hidden" == c.visibility ? !0 : a.hasAttribute("aria-hidden") && "true" == a.getAttribute("aria-hidden").toLowerCase() ? !b : !1;
};
axs.utils.isElementOrAncestorHidden = function(a) {
  return axs.utils.isElementHidden(a) ? !0 : axs.utils.parentElement(a) ? axs.utils.isElementOrAncestorHidden(axs.utils.parentElement(a)) : !1;
};
axs.utils.isInlineElement = function(a) {
  a = a.tagName.toUpperCase();
  return axs.constants.InlineElements[a];
};
axs.utils.getRoles = function(a) {
  if (!a.hasAttribute("role")) {
    return!1;
  }
  a = a.getAttribute("role").split(" ");
  for (var b = [], c = !0, d = 0;d < a.length;d++) {
    var e = a[d];
    axs.constants.ARIA_ROLES[e] ? b.push({name:e, details:axs.constants.ARIA_ROLES[e], valid:!0}) : (b.push({name:e, valid:!1}), c = !1);
  }
  return{roles:b, valid:c};
};
axs.utils.getAriaPropertyValue = function(a, b, c) {
  var d = a.replace(/^aria-/, ""), e = axs.constants.ARIA_PROPERTIES[d], d = {name:a, rawValue:b};
  if (!e) {
    return d.valid = !1, d.reason = '"' + a + '" is not a valid ARIA property', d;
  }
  e = e.valueType;
  if (!e) {
    return d.valid = !1, d.reason = '"' + a + '" is not a valid ARIA property', d;
  }
  switch(e) {
    case "idref":
      a = axs.utils.isValidIDRefValue(b, c), d.valid = a.valid, d.reason = a.reason, d.idref = a.idref;
    case "idref_list":
      a = b.split(/\s+/);
      d.valid = !0;
      for (b = 0;b < a.length;b++) {
        e = axs.utils.isValidIDRefValue(a[b], c), e.valid || (d.valid = !1), d.values ? d.values.push(e) : d.values = [e];
      }
      return d;
    case "integer":
    ;
    case "decimal":
      c = axs.utils.isValidNumber(b);
      if (!c.valid) {
        return d.valid = !1, d.reason = c.reason, d;
      }
      Math.floor(c.value) != c.value ? (d.valid = !1, d.reason = "" + b + " is not a whole integer") : (d.valid = !0, d.value = c.value);
      return d;
    case "number":
      c = axs.utils.isValidNumber(b), c.valid && (d.valid = !0, d.value = c.value);
    case "string":
      return d.valid = !0, d.value = b, d;
    case "token":
      return c = axs.utils.isValidTokenValue(a, b.toLowerCase()), c.valid ? (d.valid = !0, d.value = c.value) : (d.valid = !1, d.value = b, d.reason = c.reason), d;
    case "token_list":
      e = b.split(/\s+/);
      d.valid = !0;
      for (b = 0;b < e.length;b++) {
        c = axs.utils.isValidTokenValue(a, e[b].toLowerCase()), c.valid || (d.valid = !1, d.reason ? (d.reason = [d.reason], d.reason.push(c.reason)) : (d.reason = c.reason, d.possibleValues = c.possibleValues)), d.values ? d.values.push(c.value) : d.values = [c.value];
      }
      return d;
    case "tristate":
      return c = axs.utils.isPossibleValue(b.toLowerCase(), axs.constants.MIXED_VALUES, a), c.valid ? (d.valid = !0, d.value = c.value) : (d.valid = !1, d.value = b, d.reason = c.reason), d;
    case "boolean":
      return c = axs.utils.isValidBoolean(b), c.valid ? (d.valid = !0, d.value = c.value) : (d.valid = !1, d.value = b, d.reason = c.reason), d;
  }
  d.valid = !1;
  d.reason = "Not a valid ARIA property";
  return d;
};
axs.utils.isValidTokenValue = function(a, b) {
  var c = a.replace(/^aria-/, "");
  return axs.utils.isPossibleValue(b, axs.constants.ARIA_PROPERTIES[c].valuesSet, a);
};
axs.utils.isPossibleValue = function(a, b, c) {
  return b[a] ? {valid:!0, value:a} : {valid:!1, value:a, reason:'"' + a + '" is not a valid value for ' + c, possibleValues:Object.keys(b)};
};
axs.utils.isValidBoolean = function(a) {
  try {
    var b = JSON.parse(a);
  } catch (c) {
    b = "";
  }
  return "boolean" != typeof b ? {valid:!1, value:a, reason:'"' + a + '" is not a true/false value'} : {valid:!0, value:b};
};
axs.utils.isValidIDRefValue = function(a, b) {
  return 0 == a.length ? {valid:!0, idref:a} : b.ownerDocument.getElementById(a) ? {valid:!0, idref:a} : {valid:!1, idref:a, reason:'No element with ID "' + a + '"'};
};
axs.utils.isValidNumber = function(a) {
  try {
    var b = JSON.parse(a);
  } catch (c) {
    return{valid:!1, value:a, reason:'"' + a + '" is not a number'};
  }
  return "number" != typeof b ? {valid:!1, value:a, reason:'"' + a + '" is not a number'} : {valid:!0, value:b};
};
axs.utils.isElementImplicitlyFocusable = function(a) {
  var b = a.ownerDocument.defaultView;
  return a instanceof b.HTMLAnchorElement || a instanceof b.HTMLAreaElement ? a.hasAttribute("href") : a instanceof b.HTMLInputElement || a instanceof b.HTMLSelectElement || a instanceof b.HTMLTextAreaElement || a instanceof b.HTMLButtonElement || a instanceof b.HTMLIFrameElement ? !a.disabled : !1;
};
axs.utils.values = function(a) {
  var b = [], c;
  for (c in a) {
    a.hasOwnProperty(c) && "function" != typeof a[c] && b.push(a[c]);
  }
  return b;
};
axs.utils.namedValues = function(a) {
  var b = {}, c;
  for (c in a) {
    a.hasOwnProperty(c) && "function" != typeof a[c] && (b[c] = a[c]);
  }
  return b;
};
axs.utils.getQuerySelectorText = function(a) {
  if (null == a || "HTML" == a.tagName) {
    return "html";
  }
  if ("BODY" == a.tagName) {
    return "body";
  }
  if (a.hasAttribute) {
    if (a.id) {
      return "#" + a.id;
    }
    if (a.className) {
      for (var b = "", c = 0;c < a.classList.length;c++) {
        b += "." + a.classList[c];
      }
      var d = 0;
      if (a.parentNode) {
        for (c = 0;c < a.parentNode.children.length;c++) {
          var e = a.parentNode.children[c];
          axs.browserUtils.matchSelector(e, b) && d++;
          if (e === a) {
            break;
          }
        }
      } else {
        d = 1;
      }
      return 1 == d ? axs.utils.getQuerySelectorText(a.parentNode) + " > " + b : axs.utils.getQuerySelectorText(a.parentNode) + " > " + b + ":nth-of-type(" + d + ")";
    }
    if (a.parentNode) {
      b = a.parentNode.children;
      d = 1;
      for (c = 0;b[c] !== a;) {
        b[c].tagName == a.tagName && d++, c++;
      }
      c = "";
      "BODY" != a.parentNode.tagName && (c = axs.utils.getQuerySelectorText(a.parentNode) + " > ");
      return 1 == d ? c + a.tagName : c + a.tagName + ":nth-of-type(" + d + ")";
    }
  } else {
    if (a.selectorText) {
      return a.selectorText;
    }
  }
  return "";
};
axs.properties = {};
axs.properties.TEXT_CONTENT_XPATH = './/text()[normalize-space(.)!=""]/parent::*[name()!="script"]';
axs.properties.getFocusProperties = function(a) {
  var b = {}, c = a.getAttribute("tabindex");
  void 0 != c ? b.tabindex = {value:c, valid:!0} : axs.utils.isElementImplicitlyFocusable(a) && (b.implicitlyFocusable = {value:!0, valid:!0});
  if (0 == Object.keys(b).length) {
    return null;
  }
  var d = axs.utils.elementIsTransparent(a), e = axs.utils.elementHasZeroArea(a), f = axs.utils.elementIsOutsideScrollArea(a), g = axs.utils.overlappingElements(a);
  if (d || e || f || 0 < g.length) {
    var c = axs.utils.isElementOrAncestorHidden(a), h = {value:!1, valid:c};
    d && (h.transparent = !0);
    e && (h.zeroArea = !0);
    f && (h.outsideScrollArea = !0);
    g && 0 < g.length && (h.overlappingElements = g);
    d = {value:c, valid:c};
    c && (d.reason = axs.properties.getHiddenReason(a));
    h.hidden = d;
    b.visible = h;
  } else {
    b.visible = {value:!0, valid:!0};
  }
  return b;
};
axs.properties.getHiddenReason = function(a) {
  if (!(a && a instanceof a.ownerDocument.defaultView.HTMLElement)) {
    return null;
  }
  if (a.hasAttribute("chromevoxignoreariahidden")) {
    var b = !0
  }
  var c = window.getComputedStyle(a, null);
  return "none" == c.display ? {property:"display: none", on:a} : "hidden" == c.visibility ? {property:"visibility: hidden", on:a} : a.hasAttribute("aria-hidden") && "true" == a.getAttribute("aria-hidden").toLowerCase() && !b ? {property:"aria-hidden", on:a} : axs.properties.getHiddenReason(axs.utils.parentElement(a));
};
axs.properties.getColorProperties = function(a) {
  var b = {};
  (a = axs.properties.getContrastRatioProperties(a)) && (b.contrastRatio = a);
  return 0 == Object.keys(b).length ? null : b;
};
axs.properties.hasDirectTextDescendant = function(a) {
  for (var b = (a.nodeType == Node.DOCUMENT_NODE ? a : a.ownerDocument).evaluate(axs.properties.TEXT_CONTENT_XPATH, a, null, XPathResult.ANY_TYPE, null), c = !1, d = b.iterateNext();null != d;d = b.iterateNext()) {
    if (d === a) {
      c = !0;
      break;
    }
  }
  return c;
};
axs.properties.getContrastRatioProperties = function(a) {
  if (!axs.properties.hasDirectTextDescendant(a)) {
    return null;
  }
  var b = {}, c = window.getComputedStyle(a, null), d = axs.utils.getBgColor(c, a);
  if (!d) {
    return null;
  }
  b.backgroundColor = axs.utils.colorToString(d);
  var e = axs.utils.getFgColor(c, a, d);
  b.foregroundColor = axs.utils.colorToString(e);
  a = axs.utils.getContrastRatioForElementWithComputedStyle(c, a);
  if (!a) {
    return null;
  }
  b.value = a.toFixed(2);
  axs.utils.isLowContrast(a, c) && (b.alert = !0);
  (c = axs.utils.suggestColors(d, e, a, c)) && Object.keys(c).length && (b.suggestedColors = c);
  return b;
};
axs.properties.findTextAlternatives = function(a, b, c, d) {
  var e = c || !1;
  c = axs.utils.asElement(a);
  if (!c || !e && !d && axs.utils.isElementOrAncestorHidden(c)) {
    return null;
  }
  if (a.nodeType == Node.TEXT_NODE) {
    return c = {type:"text"}, c.text = a.textContent, c.lastWord = axs.properties.getLastWord(c.text), b.content = c, a.textContent;
  }
  a = null;
  e || (a = axs.properties.getTextFromAriaLabelledby(c, b));
  c.hasAttribute("aria-label") && (d = {type:"text"}, d.text = c.getAttribute("aria-label"), d.lastWord = axs.properties.getLastWord(d.text), a ? d.unused = !0 : e && axs.utils.elementIsHtmlControl(c) || (a = d.text), b.ariaLabel = d);
  c.hasAttribute("role") && "presentation" == c.getAttribute("role") || (a = axs.properties.getTextFromHostLanguageAttributes(c, b, a, e));
  if (e && axs.utils.elementIsHtmlControl(c)) {
    d = c.ownerDocument.defaultView;
    if (c instanceof d.HTMLInputElement) {
      var f = c;
      "text" == f.type && f.value && 0 < f.value.length && (b.controlValue = {text:f.value});
      "range" == f.type && (b.controlValue = {text:f.value});
    }
    c instanceof d.HTMLSelectElement && (b.controlValue = {text:c.value});
    b.controlValue && (d = b.controlValue, a ? d.unused = !0 : a = d.text);
  }
  if (e && axs.utils.elementIsAriaWidget(c)) {
    e = c.getAttribute("role");
    "textbox" == e && c.textContent && 0 < c.textContent.length && (b.controlValue = {text:c.textContent});
    if ("slider" == e || "spinbutton" == e) {
      c.hasAttribute("aria-valuetext") ? b.controlValue = {text:c.getAttribute("aria-valuetext")} : c.hasAttribute("aria-valuenow") && (b.controlValue = {value:c.getAttribute("aria-valuenow"), text:"" + c.getAttribute("aria-valuenow")});
    }
    if ("menu" == e) {
      var g = c.querySelectorAll("[role=menuitemcheckbox], [role=menuitemradio]");
      d = [];
      for (f = 0;f < g.length;f++) {
        "true" == g[f].getAttribute("aria-checked") && d.push(g[f]);
      }
      if (0 < d.length) {
        g = "";
        for (f = 0;f < d.length;f++) {
          g += axs.properties.findTextAlternatives(d[f], {}, !0), f < d.length - 1 && (g += ", ");
        }
        b.controlValue = {text:g};
      }
    }
    if ("combobox" == e || "select" == e) {
      b.controlValue = {text:"TODO"};
    }
    b.controlValue && (d = b.controlValue, a ? d.unused = !0 : a = d.text);
  }
  d = !0;
  c.hasAttribute("role") && (e = c.getAttribute("role"), (e = axs.constants.ARIA_ROLES[e]) && (!e.namefrom || 0 > e.namefrom.indexOf("contents")) && (d = !1));
  (e = axs.properties.getTextFromDescendantContent(c)) && d && (d = {type:"text"}, d.text = e, d.lastWord = axs.properties.getLastWord(d.text), a ? d.unused = !0 : a = e, b.content = d);
  c.hasAttribute("title") && (e = {type:"string", valid:!0}, e.text = c.getAttribute("title"), e.lastWord = axs.properties.getLastWord(e.lastWord), a ? e.unused = !0 : a = e.text, b.title = e);
  return 0 == Object.keys(b).length && null == a ? null : a;
};
axs.properties.getTextFromDescendantContent = function(a) {
  var b = a.childNodes;
  a = [];
  for (var c = 0;c < b.length;c++) {
    var d = axs.properties.findTextAlternatives(b[c], {}, !0);
    d && a.push(d.trim());
  }
  if (a.length) {
    b = "";
    for (c = 0;c < a.length;c++) {
      b = [b, a[c]].join(" ").trim();
    }
    return b;
  }
  return null;
};
axs.properties.getTextFromAriaLabelledby = function(a, b) {
  var c = null;
  if (!a.hasAttribute("aria-labelledby")) {
    return c;
  }
  for (var d = a.getAttribute("aria-labelledby").split(/\s+/), e = {valid:!0}, f = [], g = [], h = 0;h < d.length;h++) {
    var k = {type:"element"}, m = d[h];
    k.value = m;
    var l = document.getElementById(m);
    l ? (k.valid = !0, k.text = axs.properties.findTextAlternatives(l, {}, !0), k.lastWord = axs.properties.getLastWord(k.text), f.push(l.textContent.trim()), k.element = l) : (k.valid = !1, e.valid = !1, k.errorMessage = {messageKey:"noElementWithId", args:[m]});
    g.push(k);
  }
  0 < g.length && (g[g.length - 1].last = !0, e.values = g, e.text = f.join(" "), e.lastWord = axs.properties.getLastWord(e.text), c = e.text, b.ariaLabelledby = e);
  return c;
};
axs.properties.getTextFromHostLanguageAttributes = function(a, b, c, d) {
  if (axs.browserUtils.matchSelector(a, "img")) {
    if (a.hasAttribute("alt")) {
      var e = {type:"string", valid:!0};
      e.text = a.getAttribute("alt");
      c ? e.unused = !0 : c = e.text;
      b.alt = e;
    } else {
      e = {valid:!1, errorMessage:"No alt value provided"}, b.alt = e, e = a.src, "string" == typeof e && (c = e.split("/").pop(), b.filename = {text:c});
    }
  }
  if (axs.browserUtils.matchSelector(a, 'input:not([type="hidden"]):not([disabled]), select:not([disabled]), textarea:not([disabled]), button:not([disabled]), video:not([disabled])') && !d) {
    if (a.hasAttribute("id")) {
      d = document.querySelectorAll('label[for="' + a.id + '"]');
      for (var e = {}, f = [], g = [], h = 0;h < d.length;h++) {
        var k = {type:"element"}, m = d[h], l = axs.properties.findTextAlternatives(m, {}, !0);
        l && 0 < l.trim().length && (k.text = l.trim(), g.push(l.trim()));
        k.element = m;
        f.push(k);
      }
      0 < f.length && (f[f.length - 1].last = !0, e.values = f, e.text = g.join(" "), e.lastWord = axs.properties.getLastWord(e.text), c ? e.unused = !0 : c = e.text, b.labelFor = e);
    }
    d = axs.utils.parentElement(a);
    for (e = {};d;) {
      if ("label" == d.tagName.toLowerCase() && (f = d, f.control == a)) {
        e.type = "element";
        e.text = axs.properties.findTextAlternatives(f, {}, !0);
        e.lastWord = axs.properties.getLastWord(e.text);
        e.element = f;
        break;
      }
      d = axs.utils.parentElement(d);
    }
    e.text && (c ? e.unused = !0 : c = e.text, b.labelWrapped = e);
    Object.keys(b).length || (b.noLabel = !0);
  }
  return c;
};
axs.properties.getLastWord = function(a) {
  if (!a) {
    return null;
  }
  var b = a.lastIndexOf(" ") + 1, c = a.length - 10;
  return a.substring(b > c ? b : c);
};
axs.properties.getTextProperties = function(a) {
  var b = {};
  a = axs.properties.findTextAlternatives(a, b, !1, !0);
  if (0 == Object.keys(b).length) {
    if (!a) {
      return null;
    }
    b.hasProperties = !1;
  } else {
    b.hasProperties = !0;
  }
  b.computedText = a;
  b.lastWord = axs.properties.getLastWord(a);
  return b;
};
axs.properties.getAriaProperties = function(a) {
  var b = {}, c = axs.properties.getGlobalAriaProperties(a), d;
  for (d in axs.constants.ARIA_PROPERTIES) {
    var e = "aria-" + d;
    if (a.hasAttribute(e)) {
      var f = a.getAttribute(e);
      c[e] = axs.utils.getAriaPropertyValue(e, f, a);
    }
  }
  0 < Object.keys(c).length && (b.properties = axs.utils.values(c));
  f = axs.utils.getRoles(a);
  if (!f) {
    return Object.keys(b).length ? b : null;
  }
  b.roles = f;
  if (!f.valid || !f.roles) {
    return b;
  }
  for (var e = f.roles, g = 0;g < e.length;g++) {
    var h = e[g];
    if (h.details && h.details.propertiesSet) {
      for (d in h.details.propertiesSet) {
        d in c || (a.hasAttribute(d) ? (f = a.getAttribute(d), c[d] = axs.utils.getAriaPropertyValue(d, f, a), "values" in c[d] && (f = c[d].values, f[f.length - 1].isLast = !0)) : h.details.requiredPropertiesSet[d] && (c[d] = {name:d, valid:!1, reason:"Required property not set"}));
      }
    }
  }
  0 < Object.keys(c).length && (b.properties = axs.utils.values(c));
  return 0 < Object.keys(b).length ? b : null;
};
axs.properties.getGlobalAriaProperties = function(a) {
  for (var b = {}, c = 0;c < axs.constants.GLOBAL_PROPERTIES.length;c++) {
    var d = axs.constants.GLOBAL_PROPERTIES[c];
    if (a.hasAttribute(d)) {
      var e = a.getAttribute(d);
      b[d] = axs.utils.getAriaPropertyValue(d, e, a);
    }
  }
  return b;
};
axs.properties.getVideoProperties = function(a) {
  if (!axs.browserUtils.matchSelector(a, "video")) {
    return null;
  }
  var b = {};
  b.captionTracks = axs.properties.getTrackElements(a, "captions");
  b.descriptionTracks = axs.properties.getTrackElements(a, "descriptions");
  b.chapterTracks = axs.properties.getTrackElements(a, "chapters");
  return b;
};
axs.properties.getTrackElements = function(a, b) {
  var c = a.querySelectorAll("track[kind=" + b + "]"), d = {};
  if (!c.length) {
    return d.valid = !1, d.reason = {messageKey:"noTracksProvided", args:[[b]]}, d;
  }
  d.valid = !0;
  for (var e = [], f = 0;f < c.length;f++) {
    var g = {}, h = c[f].getAttribute("src"), k = c[f].getAttribute("srcLang"), m = c[f].getAttribute("label");
    h ? (g.valid = !0, g.src = h) : (g.valid = !1, g.reason = {messageKey:"noSrcProvided"});
    h = "";
    m && (h += m, k && (h += " "));
    k && (h += "(" + k + ")");
    "" == h && (h = "[[object Object]]");
    g.name = h;
    e.push(g);
  }
  d.values = e;
  return d;
};
axs.properties.getAllProperties = function(a) {
  var b = axs.utils.asElement(a);
  if (!b) {
    return{};
  }
  var c = {};
  c.ariaProperties = axs.properties.getAriaProperties(b);
  c.colorProperties = axs.properties.getColorProperties(b);
  c.focusProperties = axs.properties.getFocusProperties(b);
  c.textProperties = axs.properties.getTextProperties(a);
  c.videoProperties = axs.properties.getVideoProperties(b);
  return c;
};
axs.AuditRule = function(a) {
  for (var b = !0, c = [], d = 0;d < axs.AuditRule.requiredFields.length;d++) {
    var e = axs.AuditRule.requiredFields[d];
    e in a || (b = !1, c.push(e));
  }
  if (!b) {
    throw "Invalid spec; the following fields were not specified: " + c.join(", ") + "\n" + JSON.stringify(a);
  }
  this.name = a.name;
  this.severity = a.severity;
  this.relevantElementMatcher_ = a.relevantElementMatcher;
  this.test_ = a.test;
  this.code = a.code;
  this.heading = a.heading || "";
  this.url = a.url || "";
  this.requiresConsoleAPI = !!a.opt_requiresConsoleAPI;
};
axs.AuditRule.requiredFields = "name severity relevantElementMatcher test code heading".split(" ");
axs.AuditRule.NOT_APPLICABLE = {result:axs.constants.AuditResult.NA};
axs.AuditRule.prototype.addElement = function(a, b) {
  a.push(b);
};
axs.AuditRule.collectMatchingElements = function(a, b, c, d) {
  if (a.nodeType == Node.ELEMENT_NODE) {
    var e = a
  }
  e && b.call(null, e) && c.push(e);
  if (e) {
    var f = e.shadowRoot || e.webkitShadowRoot;
    if (f) {
      axs.AuditRule.collectMatchingElements(f, b, c, f);
      return;
    }
  }
  if (e && "content" == e.localName) {
    for (e = e.getDistributedNodes(), a = 0;a < e.length;a++) {
      axs.AuditRule.collectMatchingElements(e[a], b, c, d);
    }
  } else {
    if (e && "shadow" == e.localName) {
      a = e, d ? (d = d.olderShadowRoot || a.olderShadowRoot) && axs.AuditRule.collectMatchingElements(d, b, c, d) : console.warn("ShadowRoot not provided for", e);
    } else {
      for (e = a.firstChild;null != e;) {
        axs.AuditRule.collectMatchingElements(e, b, c, d), e = e.nextSibling;
      }
    }
  }
};
axs.AuditRule.prototype.run = function(a) {
  a = a || {};
  var b = "ignoreSelectors" in a ? a.ignoreSelectors : [], c = "maxResults" in a ? a.maxResults : null, d = [];
  axs.AuditRule.collectMatchingElements("scope" in a ? a.scope : document, this.relevantElementMatcher_, d);
  var e = [];
  if (!d.length) {
    return{result:axs.constants.AuditResult.NA};
  }
  for (a = 0;a < d.length && !(null != c && e.length >= c);a++) {
    var f = d[a], g;
    a: {
      g = f;
      for (var h = 0;h < b.length;h++) {
        if (axs.browserUtils.matchSelector(g, b[h])) {
          g = !0;
          break a;
        }
      }
      g = !1;
    }
    !g && this.test_(f) && this.addElement(e, f);
  }
  b = {result:e.length ? axs.constants.AuditResult.FAIL : axs.constants.AuditResult.PASS, elements:e};
  a < d.length && (b.resultsTruncated = !0);
  return b;
};
axs.AuditRule.specs = {};
axs.AuditRules = {};
axs.AuditRules.getRule = function(a) {
  if (!axs.AuditRules.rules) {
    axs.AuditRules.rules = {};
    for (var b in axs.AuditRule.specs) {
      var c = axs.AuditRule.specs[b], d = new axs.AuditRule(c);
      axs.AuditRules.rules[c.name] = d;
    }
  }
  return axs.AuditRules.rules[a];
};
axs.AuditResults = function() {
  this.errors_ = [];
  this.warnings_ = [];
};
goog.exportSymbol("axs.AuditResults", axs.AuditResults);
axs.AuditResults.prototype.addError = function(a) {
  "" != a && this.errors_.push(a);
};
goog.exportProperty(axs.AuditResults.prototype, "addError", axs.AuditResults.prototype.addError);
axs.AuditResults.prototype.addWarning = function(a) {
  "" != a && this.warnings_.push(a);
};
goog.exportProperty(axs.AuditResults.prototype, "addWarning", axs.AuditResults.prototype.addWarning);
axs.AuditResults.prototype.numErrors = function() {
  return this.errors_.length;
};
goog.exportProperty(axs.AuditResults.prototype, "numErrors", axs.AuditResults.prototype.numErrors);
axs.AuditResults.prototype.numWarnings = function() {
  return this.warnings_.length;
};
goog.exportProperty(axs.AuditResults.prototype, "numWarnings", axs.AuditResults.prototype.numWarnings);
axs.AuditResults.prototype.getErrors = function() {
  return this.errors_;
};
goog.exportProperty(axs.AuditResults.prototype, "getErrors", axs.AuditResults.prototype.getErrors);
axs.AuditResults.prototype.getWarnings = function() {
  return this.warnings_;
};
goog.exportProperty(axs.AuditResults.prototype, "getWarnings", axs.AuditResults.prototype.getWarnings);
axs.AuditResults.prototype.toString = function() {
  for (var a = "", b = 0;b < this.errors_.length;b++) {
    0 == b && (a += "\nErrors:\n");
    var c = this.errors_[b], a = a + (c + "\n\n");
  }
  for (b = 0;b < this.warnings_.length;b++) {
    0 == b && (a += "\nWarnings:\n"), c = this.warnings_[b], a += c + "\n\n";
  }
  return a;
};
goog.exportProperty(axs.AuditResults.prototype, "toString", axs.AuditResults.prototype.toString);
axs.Audit = {};
axs.AuditConfiguration = function() {
  this.rules_ = {};
  this.maxResults = this.auditRulesToIgnore = this.auditRulesToRun = this.scope = null;
  this.withConsoleApi = !1;
  this.showUnsupportedRulesWarning = !0;
  goog.exportProperty(this, "scope", this.scope);
  goog.exportProperty(this, "auditRulesToRun", this.auditRulesToRun);
  goog.exportProperty(this, "auditRulesToIgnore", this.auditRulesToIgnore);
  goog.exportProperty(this, "withConsoleApi", this.withConsoleApi);
  goog.exportProperty(this, "showUnsupportedRulesWarning", this.showUnsupportedRulesWarning);
};
goog.exportSymbol("axs.AuditConfiguration", axs.AuditConfiguration);
axs.AuditConfiguration.prototype = {ignoreSelectors:function(a, b) {
  a in this.rules_ || (this.rules_[a] = {});
  "ignore" in this.rules_[a] || (this.rules_[a].ignore = []);
  Array.prototype.push.call(this.rules_[a].ignore, b);
}, getIgnoreSelectors:function(a) {
  return a in this.rules_ && "ignore" in this.rules_[a] ? this.rules_[a].ignore : [];
}, setSeverity:function(a, b) {
  a in this.rules_ || (this.rules_[a] = {});
  this.rules_[a].severity = b;
}, getSeverity:function(a) {
  return a in this.rules_ && "severity" in this.rules_[a] ? this.rules_[a].severity : null;
}};
goog.exportProperty(axs.AuditConfiguration.prototype, "ignoreSelectors", axs.AuditConfiguration.prototype.ignoreSelectors);
goog.exportProperty(axs.AuditConfiguration.prototype, "getIgnoreSelectors", axs.AuditConfiguration.prototype.getIgnoreSelectors);
axs.Audit.unsupportedRulesWarningShown = !1;
axs.Audit.getRulesCannotRun = function(a) {
  return a.withConsoleApi ? [] : Object.keys(axs.AuditRule.specs).filter(function(a) {
    return axs.AuditRules.getRule(a).requiresConsoleAPI;
  }).map(function(a) {
    return axs.AuditRules.getRule(a).code;
  });
};
axs.Audit.run = function(a) {
  a = a || new axs.AuditConfiguration;
  var b = a.withConsoleApi, c = [], d;
  d = a.auditRulesToRun && 0 < a.auditRulesToRun.length ? a.auditRulesToRun : Object.keys(axs.AuditRule.specs);
  if (a.auditRulesToIgnore) {
    for (var e = 0;e < a.auditRulesToIgnore.length;e++) {
      var f = a.auditRulesToIgnore[e];
      0 > d.indexOf(f) || d.splice(d.indexOf(f), 1);
    }
  }
  !axs.Audit.unsupportedRulesWarningShown && a.showUnsupportedRulesWarning && (e = axs.Audit.getRulesCannotRun(a), 0 < e.length && (console.warn("Some rules cannot be checked using the axs.Audit.run() method call. Use the Chrome plugin to check these rules: " + e.join(", ")), console.warn("To remove this message, pass an AuditConfiguration object to axs.Audit.run() and set configuration.showUnsupportedRulesWarning = false.")), axs.Audit.unsupportedRulesWarningShown = !0);
  for (e = 0;e < d.length;e++) {
    var f = d[e], g = axs.AuditRules.getRule(f);
    if (g && !g.disabled && (b || !g.requiresConsoleAPI)) {
      var h = {}, k = a.getIgnoreSelectors(g.name);
      if (0 < k.length || a.scope) {
        h.ignoreSelectors = k;
      }
      a.scope && (h.scope = a.scope);
      a.maxResults && (h.maxResults = a.maxResults);
      h = g.run.call(g, h);
      g = axs.utils.namedValues(g);
      g.severity = a.getSeverity(f) || g.severity;
      h.rule = g;
      c.push(h);
    }
  }
  return c;
};
goog.exportSymbol("axs.Audit.run", axs.Audit.run);
axs.Audit.auditResults = function(a) {
  for (var b = new axs.AuditResults, c = 0;c < a.length;c++) {
    var d = a[c];
    d.result == axs.constants.AuditResult.FAIL && (d.rule.severity == axs.constants.Severity.SEVERE ? b.addError(axs.Audit.accessibilityErrorMessage(d)) : b.addWarning(axs.Audit.accessibilityErrorMessage(d)));
  }
  return b;
};
goog.exportSymbol("axs.Audit.auditResults", axs.Audit.auditResults);
axs.Audit.createReport = function(a, b) {
  var c;
  c = "*** Begin accessibility audit results ***\nAn accessibility audit found " + axs.Audit.auditResults(a).toString();
  b && (c += "\nFor more information, please see ", c += b);
  return c += "\n*** End accessibility audit results ***";
};
goog.exportSymbol("axs.Audit.createReport", axs.Audit.createReport);
axs.Audit.accessibilityErrorMessage = function(a) {
  for (var b = a.rule.severity == axs.constants.Severity.SEVERE ? "Error: " : "Warning: ", b = b + (a.rule.code + " (" + a.rule.heading + ") failed on the following " + (1 == a.elements.length ? "element" : "elements")), b = 1 == a.elements.length ? b + ":" : b + (" (1 - " + Math.min(5, a.elements.length) + " of " + a.elements.length + "):"), c = Math.min(a.elements.length, 5), d = 0;d < c;d++) {
    var e = a.elements[d], b = b + "\n";
    try {
      b += axs.utils.getQuerySelectorText(e);
    } catch (f) {
      b += " tagName:" + e.tagName, b += " id:" + e.id;
    }
  }
  "" != a.rule.url && (b += "\nSee " + a.rule.url + " for more information.");
  return b;
};
goog.exportSymbol("axs.Audit.accessibilityErrorMessage", axs.Audit.accessibilityErrorMessage);
axs.AuditRule.specs.audioWithoutControls = {name:"audioWithoutControls", heading:"Audio elements should have controls", url:"", severity:axs.constants.Severity.WARNING, relevantElementMatcher:function(a) {
  return axs.browserUtils.matchSelector(a, "audio[autoplay]");
}, test:function(a) {
  return!a.querySelectorAll("[controls]").length && 3 < a.duration;
}, code:"AX_AUDIO_01"};
axs.AuditRule.specs.badAriaAttributeValue = {name:"badAriaAttributeValue", heading:"ARIA state and property values must be valid", url:"", severity:axs.constants.Severity.SEVERE, relevantElementMatcher:function(a) {
  var b = "", c;
  for (c in axs.constants.ARIA_PROPERTIES) {
    b += "[aria-" + c + "],";
  }
  b = b.substring(0, b.length - 1);
  return axs.browserUtils.matchSelector(a, b);
}, test:function(a) {
  for (var b in axs.constants.ARIA_PROPERTIES) {
    var c = "aria-" + b;
    if (a.hasAttribute(c)) {
      var d = a.getAttribute(c);
      if (!axs.utils.getAriaPropertyValue(c, d, a).valid) {
        return!0;
      }
    }
  }
  return!1;
}, code:"AX_ARIA_04"};
axs.AuditRule.specs.badAriaRole = {name:"badAriaRole", heading:"Elements with ARIA roles must use a valid, non-abstract ARIA role", url:"https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_aria_01--elements-with-aria-roles-must-use-a-valid-non-abstract-aria-role", severity:axs.constants.Severity.SEVERE, relevantElementMatcher:function(a) {
  return axs.browserUtils.matchSelector(a, "[role]");
}, test:function(a) {
  return!axs.utils.getRoles(a).valid;
}, code:"AX_ARIA_01"};
axs.AuditRule.specs.controlsWithoutLabel = {name:"controlsWithoutLabel", heading:"Controls and media elements should have labels", url:"https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_text_01--controls-and-media-elements-should-have-labels", severity:axs.constants.Severity.SEVERE, relevantElementMatcher:function(a) {
  if (!axs.browserUtils.matchSelector(a, 'input:not([type="hidden"]):not([disabled]), select:not([disabled]), textarea:not([disabled]), button:not([disabled]), video:not([disabled])')) {
    return!1;
  }
  if (0 <= a.tabIndex) {
    return!0;
  }
  for (a = axs.utils.parentElement(a);null != a;a = axs.utils.parentElement(a)) {
    if (axs.utils.elementIsAriaWidget(a)) {
      return!1;
    }
  }
  return!0;
}, test:function(a) {
  return axs.utils.isElementOrAncestorHidden(a) || "input" == a.tagName.toLowerCase() && "button" == a.type && a.value.length || "button" == a.tagName.toLowerCase() && a.textContent.replace(/^\s+|\s+$/g, "").length ? !1 : axs.utils.hasLabel(a) ? !1 : !0;
}, code:"AX_TEXT_01", ruleName:"Controls and media elements should have labels"};
axs.AuditRule.specs.focusableElementNotVisibleAndNotAriaHidden = {name:"focusableElementNotVisibleAndNotAriaHidden", heading:"These elements are focusable but either invisible or obscured by another element", url:"https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_focus_01--these-elements-are-focusable-but-either-invisible-or-obscured-by-another-element", severity:axs.constants.Severity.WARNING, relevantElementMatcher:function(a) {
  if (!axs.browserUtils.matchSelector(a, axs.utils.FOCUSABLE_ELEMENTS_SELECTOR)) {
    return!1;
  }
  if (0 <= a.tabIndex) {
    return!0;
  }
  for (a = axs.utils.parentElement(a);null != a;a = axs.utils.parentElement(a)) {
    if (axs.utils.elementIsAriaWidget(a)) {
      return!1;
    }
  }
  return!0;
}, test:function(a) {
  if (axs.utils.isElementOrAncestorHidden(a)) {
    return!1;
  }
  a.focus();
  return!axs.utils.elementIsVisible(a);
}, code:"AX_FOCUS_01"};
axs.AuditRule.specs.imagesWithoutAltText = {name:"imagesWithoutAltText", heading:"Images should have an alt attribute", url:"https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_text_02--images-should-have-an-alt-attribute-unless-they-have-an-aria-role-of-presentation", severity:axs.constants.Severity.WARNING, relevantElementMatcher:function(a) {
  return axs.browserUtils.matchSelector(a, "img") && !axs.utils.isElementOrAncestorHidden(a);
}, test:function(a) {
  return!a.hasAttribute("alt") && "presentation" != a.getAttribute("role");
}, code:"AX_TEXT_02"};
axs.AuditRule.specs.linkWithUnclearPurpose = {name:"linkWithUnclearPurpose", heading:"The purpose of each link should be clear from the link text", url:"", severity:axs.constants.Severity.WARNING, relevantElementMatcher:function(a) {
  return axs.browserUtils.matchSelector(a, "a");
}, test:function(a) {
  return/^\s*click\s*here\s*[^a-z]?$/i.test(a.textContent);
}, code:"AX_TITLE_01"};
axs.AuditRule.specs.lowContrastElements = {name:"lowContrastElements", heading:"Text elements should have a reasonable contrast ratio", url:"https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_color_01--text-elements-should-have-a-reasonable-contrast-ratio", severity:axs.constants.Severity.WARNING, relevantElementMatcher:function(a) {
  return axs.properties.hasDirectTextDescendant(a);
}, test:function(a) {
  var b = window.getComputedStyle(a, null);
  return(a = axs.utils.getContrastRatioForElementWithComputedStyle(b, a)) && axs.utils.isLowContrast(a, b);
}, code:"AX_COLOR_01"};
axs.AuditRule.specs.mainRoleOnInappropriateElement = {name:"mainRoleOnInappropriateElement", heading:"role=main should only appear on significant elements", url:"", severity:axs.constants.Severity.WARNING, relevantElementMatcher:function(a) {
  return axs.browserUtils.matchSelector(a, "[role~=main]");
}, test:function(a) {
  if (axs.utils.isInlineElement(a)) {
    return!0;
  }
  a = axs.properties.getTextFromDescendantContent(a);
  return!a || 50 > a.length ? !0 : !1;
}, code:"AX_ARIA_04"};
axs.AuditRule.specs.elementsWithMeaningfulBackgroundImage = {name:"elementsWithMeaningfulBackgroundImage", severity:axs.constants.Severity.WARNING, relevantElementMatcher:function(a) {
  return!axs.utils.isElementOrAncestorHidden(a);
}, heading:"Meaningful images should not be used in element backgrounds", url:"https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_image_01--meaningful-images-should-not-be-used-in-element-backgrounds", test:function(a) {
  if (a.textContent && 0 < a.textContent.length) {
    return!1;
  }
  a = window.getComputedStyle(a, null);
  var b = a.backgroundImage;
  if (!b || "undefined" === b || "none" === b || 0 != b.indexOf("url")) {
    return!1;
  }
  b = parseInt(a.width, 10);
  a = parseInt(a.height, 10);
  return 150 > b && 150 > a;
}, code:"AX_IMAGE_01"};
axs.AuditRule.specs.nonExistentAriaLabelledbyElement = {name:"nonExistentAriaLabelledbyElement", heading:"aria-labelledby attributes should refer to an element which exists in the DOM", url:"https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_aria_02--aria-labelledby-attributes-should-refer-to-an-element-which-exists-in-the-dom", severity:axs.constants.Severity.WARNING, relevantElementMatcher:function(a) {
  return axs.browserUtils.matchSelector(a, "[aria-labelledby]");
}, test:function(a) {
  a = a.getAttribute("aria-labelledby").split(/\s+/);
  for (var b = 0;b < a.length;b++) {
    if (!document.getElementById(a[b])) {
      return!0;
    }
  }
  return!1;
}, code:"AX_ARIA_02"};
axs.AuditRule.specs.pageWithoutTitle = {name:"pageWithoutTitle", heading:"The web page should have a title that describes topic or purpose", url:"", severity:axs.constants.Severity.WARNING, relevantElementMatcher:function(a) {
  return "html" == a.tagName.toLowerCase();
}, test:function(a) {
  a = a.querySelector("head");
  return a ? (a = a.querySelector("title")) ? !a.textContent : !0 : !0;
}, code:"AX_TITLE_01"};
axs.AuditRule.specs.requiredAriaAttributeMissing = {name:"requiredAriaAttributeMissing", heading:"Elements with ARIA roles must have all required attributes for that role", url:"https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_aria_03--elements-with-aria-roles-must-have-all-required-attributes-for-that-role", severity:axs.constants.Severity.SEVERE, relevantElementMatcher:function(a) {
  return axs.browserUtils.matchSelector(a, "[role]");
}, test:function(a) {
  var b = axs.utils.getRoles(a);
  if (!b.valid) {
    return!1;
  }
  for (var c = 0;c < b.roles.length;c++) {
    var d = b.roles[c].details.requiredPropertiesSet, e;
    for (e in d) {
      if (d = e.replace(/^aria-/, ""), !("defaultValue" in axs.constants.ARIA_PROPERTIES[d] || a.hasAttribute(e))) {
        return!0;
      }
    }
  }
}, code:"AX_ARIA_03"};
axs.AuditRule.specs.unfocusableElementsWithOnClick = {name:"unfocusableElementsWithOnClick", heading:"Elements with onclick handlers must be focusable", url:"https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_focus_02--elements-with-onclick-handlers-must-be-focusable", severity:axs.constants.Severity.WARNING, opt_requiresConsoleAPI:!0, relevantElementMatcher:function(a) {
  return a instanceof a.ownerDocument.defaultView.HTMLBodyElement || axs.utils.isElementOrAncestorHidden(a) ? !1 : "click" in getEventListeners(a) ? !0 : !1;
}, test:function(a) {
  return!a.hasAttribute("tabindex") && !axs.utils.isElementImplicitlyFocusable(a) && !a.disabled;
}, code:"AX_FOCUS_02"};
axs.AuditRule.specs.videoWithoutCaptions = {name:"videoWithoutCaptions", heading:"Video elements should use <track> elements to provide captions", url:"https://github.com/GoogleChrome/accessibility-developer-tools/wiki/Audit-Rules#-ax_video_01--video-elements-should-use-track-elements-to-provide-captions", severity:axs.constants.Severity.WARNING, relevantElementMatcher:function(a) {
  return axs.browserUtils.matchSelector(a, "video");
}, test:function(a) {
  return!a.querySelectorAll("track[kind=captions]").length;
}, code:"AX_VIDEO_01"};



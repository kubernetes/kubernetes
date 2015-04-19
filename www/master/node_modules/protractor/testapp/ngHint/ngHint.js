(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
'use strict';

// Create pipe for all hint messages from different modules
window.angular.hint = require('angular-hint-log');

// Load angular hint modules
require('angular-hint-controllers');
require('angular-hint-directives');
//require('angular-hint-dom');
require('angular-hint-events');
//require('angular-hint-interpolation');
require('angular-hint-modules');
require('angular-hint-scopes');

// List of all possible modules
// The default ng-hint behavior loads all modules
var allModules = [
  'ngHintControllers',
  'ngHintDirectives',
//  'ngHintDom',
  'ngHintEvents',
//  'ngHintInterpolation',
  'ngHintModules',
  'ngHintScopes'
];

var SEVERITY_WARNING = 2;

// Determine whether this run is by protractor.
// If protractor is running, the bootstrap will already be deferred.
// In this case `resumeBootstrap` should be patched to load the hint modules.
if (window.name === 'NG_DEFER_BOOTSTRAP!') {
  var originalResumeBootstrap;
  Object.defineProperty(angular, 'resumeBootstrap', {
    get: function() {
      return function(modules) {
        return originalResumeBootstrap.call(angular, modules.concat(loadModules()));
      };
    },
    set: function(resumeBootstrap) {
      originalResumeBootstrap = resumeBootstrap;
    }
  });
}
//If this is not a test, defer bootstrapping
else {
  window.name = 'NG_DEFER_BOOTSTRAP!';

  // determine which modules to load and resume bootstrap
  document.addEventListener('DOMContentLoaded', maybeBootstrap);
}

function maybeBootstrap() {
  // we don't know if angular is loaded
  if (!angular.resumeBootstrap) {
    return setTimeout(maybeBootstrap, 1);
  }

  var modules = loadModules();
  angular.resumeBootstrap(modules);
}

function loadModules() {
  var modules = [], elt;

  if ((elt = document.querySelector('[ng-hint-include]'))) {
    modules = hintModulesFromElement(elt);
  } else if (elt = document.querySelector('[ng-hint-exclude]')) {
    modules = excludeModules(hintModulesFromElement(elt));
  } else if (document.querySelector('[ng-hint]')) {
    modules = allModules;
  } else {
    angular.hint.logMessage('General', 'ngHint is included on the page, but is not active because ' +
      'there is no `ng-hint` attribute present', SEVERITY_WARNING);
  }
  return modules;
}

function excludeModules(modulesToExclude) {
  return allModules.filter(function(module) {
    return modulesToExclude.indexOf(module) === -1;
  });
}

function hintModulesFromElement (elt) {
  var selectedModules = (elt.attributes['ng-hint-include'] ||
    elt.attributes['ng-hint-exclude']).value.split(' ');

  return selectedModules.map(hintModuleName).filter(function (name) {
    return (allModules.indexOf(name) > -1) ||
      angular.hint.logMessage('General', 'Module ' + name + ' could not be found', SEVERITY_WARNING);
  });
}

function hintModuleName(name) {
  return 'ngHint' + title(name);
}

function title(str) {
  return str[0].toUpperCase() + str.substr(1);
}

var LEVELS = [
  'error',
  'warning',
  'suggestion'
];

function flush() {
  var log = angular.hint.flush(),
      groups = Object.keys(log);

  groups.forEach(function (groupName) {
    var group = log[groupName];
    var header = 'Angular Hint: ' + groupName;

    console.groupCollapsed ?
        console.groupCollapsed(header) :
        console.log(header);

    LEVELS.forEach(function (level) {
      group[level] && logGroup(group[level], title(level));
    });
    console.groupEnd && console.groupEnd();
  });
}

setInterval(flush, 2)

angular.hint.onMessage = function () {
  setTimeout(flush, 2);
};

function logGroup(group, type) {
  console.group ? console.group(type) : console.log(type);
  for(var i = 0, ii = group.length; i < ii; i++) {
    console.log(group[i]);
  }
  console.group && console.groupEnd();
}

},{"angular-hint-controllers":2,"angular-hint-directives":3,"angular-hint-events":38,"angular-hint-log":47,"angular-hint-modules":48,"angular-hint-scopes":66}],2:[function(require,module,exports){
'use strict';

var nameToControllerMatch = {},
  controllers = {},
  hintLog = angular.hint = require('angular-hint-log'),
  MODULE_NAME = 'Controllers',
  SEVERITY_ERROR = 1,
  SEVERITY_WARNING = 2;

/**
* Decorates $controller with a patching function to
* log a message if the controller is instantiated on the window
*/
angular.module('ngHintControllers', []).
  config(function ($provide) {
    $provide.decorator('$controller', function($delegate) {
        return function(ctrl, locals) {
          //If the controller name is passed, find the controller than matches it
          if(typeof ctrl === 'string') {
            if(nameToControllerMatch[ctrl]) {
              ctrl = nameToControllerMatch[ctrl];
            } else {
              //If the controller function cannot be found, check for it on the window
              checkUppercaseName(ctrl);
              checkControllerInName(ctrl);
              ctrl = window[ctrl] || ctrl;
              if(typeof ctrl === 'string') {
                throw new Error('The controller function for ' + ctrl + ' could not be found.' +
                  ' Is the function registered under that name?');
              }
            }
          }
          locals = locals || {};
          //If the controller is not in the list of already registered controllers
          //and it is not connected to the local scope, it must be instantiated on the window
          if(!controllers[ctrl] && (!locals.$scope || !locals.$scope[ctrl]) &&
              ctrl.toString().indexOf('@name ngModel.NgModelController#$render') === -1 &&
              ctrl.toString().indexOf('@name form.FormController') === -1) {
            if(angular.version.minor <= 2) {
              hintLog.logMessage(MODULE_NAME, 'It is against Angular best practices to ' +
                'instantiate a controller on the window. This behavior is deprecated in Angular' +
                ' 1.3.0', SEVERITY_WARNING);
            } else {
              hintLog.logMessage(MODULE_NAME, 'Global instantiation of controllers was deprecated' +
                ' in Angular 1.3.0. Define the controller on a module.', SEVERITY_ERROR);
            }
          }
          var ctrlInstance = $delegate.apply(this, [ctrl, locals]);
          return ctrlInstance;
        };
    });
});

/**
* Save details of the controllers as they are instantiated
* for use in decoration.
* Hint about the best practices for naming controllers.
*/
var originalModule = angular.module;

function checkUppercaseName(controllerName) {
  var firstLetter = controllerName.charAt(0);
  if(firstLetter !== firstLetter.toUpperCase() && firstLetter === firstLetter.toLowerCase()) {
    hintLog.logMessage(MODULE_NAME, 'The best practice is to name controllers with an' +
      ' uppercase first letter. Check the name of \'' + controllerName + '\'.', SEVERITY_WARNING);
  }
}

function checkControllerInName(controllerName) {
  var splitName = controllerName.split('Controller');
  if(splitName.length === 1 || splitName[splitName.length - 1] !== '') {
    hintLog.logMessage(MODULE_NAME, 'The best practice is to name controllers ending with ' +
      '\'Controller\'. Check the name of \'' + controllerName + '\'.', SEVERITY_WARNING);
  }
}

angular.module = function() {
  var module = originalModule.apply(this, arguments),
    originalController = module.controller;
  module.controller = function(controllerName, controllerConstructor) {
    nameToControllerMatch[controllerName] = controllerConstructor;
    controllers[controllerConstructor] = controllerConstructor;
    checkUppercaseName(controllerName);
    checkControllerInName(controllerName);
    return originalController.apply(this, arguments);
  };
  return module;
};

},{"angular-hint-log":47}],3:[function(require,module,exports){
'use strict';

var ddLibData = require('./lib/ddLib-data');

var RESTRICT_REGEXP = /restrict\s*:\s*['"](.+?)['"]/;
var customDirectives = [];
var dasherize = require('dasherize');
var search = require('./lib/search');
var checkPrelimErrors = require('./lib/checkPrelimErrors');
var getKeysAndValues = require('./lib/getKeysAndValues');
var defaultDirectives = ddLibData.directiveTypes['angular-default-directives'].directives;
var htmlDirectives = ddLibData.directiveTypes['html-directives'].directives;

angular.module('ngHintDirectives', [])
  .config(['$provide', function($provide) {
    $provide.decorator('$compile', ['$delegate', function($delegate) {
      return function(elem) {
        elem = angular.element(elem);
        for(var i = 0, length = elem.length; i < length; i+=2) {
          if(elem[i].getElementsByTagName) {
            var toSend = Array.prototype.slice.call(elem[i].getElementsByTagName('*'));
            search(toSend, customDirectives);
          }
        }
        return $delegate.apply(this, arguments);
      };
    }]);
  }]);

function supportObject(directiveObject) {
  if(typeof directiveObject === 'object') {
    var keys = Object.keys(directiveObject);
    for(var i = keys.length - 1; i >= 0; i--) {
      if(typeof directiveObject[keys[i]] === 'function') {
        return directiveObject[keys[i]];
      }
    }
  }
  return function() {};
}

var originalAngularModule = angular.module;
angular.module = function() {
  var module = originalAngularModule.apply(this, arguments);
  var originalDirective = module.directive;
  module.directive = function(directiveName, directiveFactory) {
    directiveFactory = directiveFactory || supportObject(directiveName);
    directiveName = typeof directiveName === 'string' ? directiveName : Object.keys(directiveName)[0];

    var originalDirectiveFactory = typeof directiveFactory === 'function' ? directiveFactory :
        directiveFactory[directiveFactory.length - 1];

    var factoryStr = originalDirectiveFactory.toString();

    checkPrelimErrors(directiveName, factoryStr);

    var pairs = getKeysAndValues(factoryStr);
    pairs.map(function(pair) {
      customDirectives.push(pair);
    });

    var matchRestrict = factoryStr.match(RESTRICT_REGEXP);
    var restrict = (matchRestrict && matchRestrict[1]) || 'A';
    var directive = {
      directiveName: directiveName,
      restrict: restrict,
      require: pairs
    };

    customDirectives.push(directive);

    return originalDirective.apply(this, arguments);
  };
  return module;
};

},{"./lib/checkPrelimErrors":16,"./lib/ddLib-data":17,"./lib/getKeysAndValues":24,"./lib/search":32,"dasherize":34}],4:[function(require,module,exports){
/**
 *@param s: first string to compare
 *@param t: second string to compare
 *
 *@description:
 *Checks to see if two strings are similiar enough to even bother checking the Levenshtein Distance.
 */
module.exports = function(s, t) {
  var strMap = {}, similarities = 0, STRICTNESS = 0.66;
  if(Math.abs(s.length - t.length) > 3) {
    return false;
  }
  s.split('').forEach(function(x){strMap[x] = x;});
  for (var i = t.length - 1; i >= 0; i--) {
    similarities = strMap[t.charAt(i)] ? similarities + 1 : similarities;
  }
  return similarities >= t.length * STRICTNESS;
};

},{}],5:[function(require,module,exports){
var ddLibData = require('./ddLib-data');

/**
 *@param attribute: attribute name as string e.g. 'ng-click', 'width', 'src', etc.
 *@param options: {} options object from beginSearch.
 *
 *@description attribute exsistance in the types of directives/attibutes (html, angular core, and
 * angular custom) and checks the restrict property of values matches its use.
 *
 *@return {} with attribute exsistance and wrong use e.g. restrict property set to elements only.
 **/
module.exports = function(attribute, options) {
  var anyTrue = false,
      wrongUse = '',
      directive,
      restrictProp;

  options.directiveTypes.forEach(function(dirType) {
    var isTag = attribute.charAt(0) === '*';
    var isCustomDir = dirType === 'angular-custom-directives';
    if(!isTag) {
      directive = ddLibData.directiveTypes[dirType].directives[attribute] || '';
      restrictProp = directive.restrict || directive;
      if(restrictProp) {
        if(restrictProp.indexOf('E') > -1 && restrictProp.indexOf('A') < 0) {
          wrongUse = 'element';
        }
        if(restrictProp.indexOf('C') > -1 && restrictProp.indexOf('A') < 0) {
          wrongUse = (wrongUse) ? 'element and class' : 'class';
        }
        anyTrue = anyTrue || true;
      }
    }
    else if(isTag && isCustomDir){
      directive = ddLibData.directiveTypes[dirType].directives[attribute.substring(1)] || '';
      restrictProp = directive.restrict || directive;
      anyTrue = anyTrue || true;
      if(restrictProp && restrictProp.indexOf('A') > -1 && restrictProp.indexOf('E') < 0) {
        wrongUse = 'attribute';
      }
    }
  });
  var typeError = wrongUse? 'wronguse' : '' || !anyTrue ? 'nonexsisting' : '' || '';
  return {exsists: anyTrue, wrongUse: wrongUse, typeError: typeError};
};

},{"./ddLib-data":17}],6:[function(require,module,exports){
var ddLibData = require('./ddLib-data'),
  SEVERITY_ERROR = 1;

module.exports = function(info, id, type) {
  var message = ddLibData.directiveTypes[info.directiveType].message + type + ' element' + id + '. ';
  var error = info.error;
  error = (error.charAt(0) === '*') ? error.substring(1): error;
  message += 'Found deprecated directive "' + error + '". Use an alternative solution.';
  return [message, SEVERITY_ERROR];
};

},{"./ddLib-data":17}],7:[function(require,module,exports){
var SEVERITY_ERROR = 1;

module.exports = function(info, id, type) {
  var missingLength = info.missing.length;
  var s = missingLength === 1 ? ' ' : 's ';
  var waswere = missingLength === 1 ? 'is ' : 'are ';
  var missing = '';
  info.missing.forEach(function(str){
    missing += '"' + str + '",';
  });
  missing = '[' + missing.substring(0, missing.length-1) + '] ';
  var message = 'Attribute' + s + missing + waswere + 'missing in ' + type + ' element' + id + '.';
  return [message, SEVERITY_ERROR];
};

},{}],8:[function(require,module,exports){
var isMutExclusiveDir = require('./isMutExclusiveDir'),
  SEVERITY_ERROR = 1;

module.exports = function(info, id, type) {
  var pair = isMutExclusiveDir(info.error);
  var message = 'Angular attributes "'+info.error+'" and "'+pair+'" in '+type+ ' element'+id+
    ' should not be attributes together on the same HTML element';
  return [message, SEVERITY_ERROR];
};

},{"./isMutExclusiveDir":29}],9:[function(require,module,exports){
var hintLog = require('angular-hint-log'),
  MODULE_NAME = 'Directives',
  SEVERITY_SUGGESTION = 3;

module.exports = function(directiveName) {
  var message = 'Directive "'+directiveName+'" should have proper namespace try adding a prefix'+
    ' and/or using camelcase.';
  var domElement = '<'+directiveName+'> </'+directiveName+'>';
  hintLog.logMessage(MODULE_NAME, message, SEVERITY_SUGGESTION);
};

},{"angular-hint-log":47}],10:[function(require,module,exports){
var SEVERITY_SUGGESTION = 3;

module.exports = function(info, id, type) {
  var ngDir = 'ng-' + info.error.substring(2),
    message = 'Use Angular version of "' + info.error + '" in ' + type + ' element' + id +
      '. Try: "' + ngDir + '"';
  return [message, SEVERITY_SUGGESTION];
};

},{}],11:[function(require,module,exports){
var SEVERITY_ERROR = 1;
module.exports = function(info, id, type) {
  var message = 'ngRepeat in '+type+' element'+id+' was used incorrectly. '+info.suggestion;
  return [message, SEVERITY_ERROR];
};

},{}],12:[function(require,module,exports){
var ddLibData = require('./ddLib-data'),
  SEVERITY_ERROR = 1;

module.exports = function(info, id, type) {
  var message = ddLibData.directiveTypes[info.directiveType].message + type + ' element' + id + '. ';
  var error = (info.error.charAt(0) === '*') ? info.error.substring(1): info.error;
  message += 'Found incorrect attribute "' + error + '" try "' + info.match + '".';
  return [message, SEVERITY_ERROR];
};

},{"./ddLib-data":17}],13:[function(require,module,exports){
var hintLog = angular.hint = require('angular-hint-log'),
  MODULE_NAME = 'Directives',
  SEVERITY_ERROR = 1;

module.exports = function(directiveName) {
  var message = 'The use of "replace" in directive factories is deprecated,'+
    ' and it was found in "' + directiveName + '".';
  var domElement = '<' + directiveName + '> </' + directiveName + '>';
  hintLog.logMessage(MODULE_NAME, message, SEVERITY_ERROR);
};

},{"angular-hint-log":47}],14:[function(require,module,exports){
var ddLibData = require('./ddLib-data'),
  SEVERITY_ERROR = 1;

module.exports = function(info, id, type) {
  var message = ddLibData.directiveTypes[info.directiveType].message + type + ' element' +
    id + '. ',
    error = (info.error.charAt(0) === '*') ? info.error.substring(1): info.error,
    aecmType = (info.wrongUse.indexOf('attribute') > -1)? 'Element' : 'Attribute';
  message += aecmType + ' name "' + error + '" is reserved for ' + info.wrongUse + ' names only.';
  return [message, SEVERITY_ERROR];
};

},{"./ddLib-data":17}],15:[function(require,module,exports){

module.exports = function(attrVal){
  var suggestion,
      error = false,
      TRACK_REGEXP = /track\s+by\s+\S*/,
      FILTER_REGEXP = /filter\s*:\s*\w+(?:\.\w+)*/;
  var trackMatch = attrVal.match(TRACK_REGEXP);
  var filterMatch = attrVal.match(FILTER_REGEXP);
  var breakIndex = attrVal.indexOf('|') > -1 ? attrVal.indexOf('|') : Infinity;

  if(!trackMatch && filterMatch && breakIndex === Infinity) {
    return 'Try: " | '+filterMatch[0]+'"';
  }

  if(trackMatch && filterMatch) {
    var trackInd = attrVal.indexOf(trackMatch[0]);
    var filterInd = attrVal.indexOf(filterMatch[0]);
    if(!(breakIndex < filterInd && filterInd < trackInd)) {
      return 'Try: " | '+filterMatch[0]+' '+trackMatch[0]+'"';
    }
  }
}
},{}],16:[function(require,module,exports){
var hasNameSpace = require('./hasNameSpace');
var buildNameSpace = require('./buildNameSpace');
var hasReplaceOption = require('./hasReplaceOption');
var buildReplaceOption = require('./buildReplaceOption');

module.exports = function(dirName, dirFacStr) {
  if (!hasNameSpace(dirName)) {
    buildNameSpace(dirName);
  }
  if (hasReplaceOption(dirFacStr)) {
    buildReplaceOption(dirName);
  }
};

},{"./buildNameSpace":9,"./buildReplaceOption":13,"./hasNameSpace":27,"./hasReplaceOption":28}],17:[function(require,module,exports){
module.exports = {
  directiveTypes : {
    'html-directives': {
      message: 'There was an HTML error in ',
      directives: {
        'abbr' : 'A',
        'accept': 'A',
        'accesskey': 'A',
        'action': 'A',
        'align': 'A',
        'alt': 'A',
        'async': 'A',
        'background': 'A',
        'bgcolor': 'A',
        'border': 'A',
        'cellpadding': 'A',
        'char': 'A',
        'charoff': 'A',
        'charset': 'A',
        'checked': 'A',
        'cite': 'A',
        'class': 'A',
        'classid': 'A',
        'code': 'A',
        'codebase': 'A',
        'color': 'A',
        'cols': 'A',
        'colspan': 'A',
        'content': 'A',
        'data': 'A',
        'defer': 'A',
        'dir': 'A',
        'face': 'A',
        'for': 'A',
        'frame': 'A',
        'frameborder': 'A',
        'headers': 'A',
        'height': 'A',
        'http-equiv': 'A',
        'href': 'A',
        'id': 'A',
        'label': 'A',
        'lang': 'A',
        'language': 'A',
        'link': 'A',
        'marginheight': 'A',
        'marginwidth': 'A',
        'maxlength': 'A',
        'media': 'A',
        'multiple': 'A',
        'name': 'A',
        'object': '!A',
        'onblur': '!A',
        'onchange': '!A',
        'onclick': '!A',
        'onfocus': '!A',
        'onkeydown': '!A',
        'onkeypress': '!A',
        'onkeyup': '!A',
        'onload': '!A',
        'onmousedown': '!A',
        'onmousemove': '!A',
        'onmouseout': '!A',
        'onmouseover': '!A',
        'onmouseup': '!A',
        'onreset': '!A',
        'onselect': '!A',
        'onsubmit': '!A',
        'property': 'A',
        'readonly': 'A',
        'rel': 'A',
        'rev': 'A',
        'role': 'A',
        'rows': 'A',
        'rowspan': 'A',
        'size': 'A',
        'span': 'EA',
        'src': 'A',
        'start': 'A',
        'style': 'A',
        'text': 'A',
        'target': 'A',
        'title': 'A',
        'type': 'A',
        'value': 'A',
        'width': 'A'
      }
    },
    'angular-default-directives': {
      message: 'There was an AngularJS error in ',
      directives: {
        'count': 'A',
        'min': 'A',
        'max': 'A',
        'ng-app': 'A',
        'ng-bind': 'A',
        'ng-bindhtml': 'A',
        'ng-bindtemplate': 'A',
        'ng-blur': 'A',
        'ng-change': 'A',
        'ng-checked': 'A',
        'ng-class': 'A',
        'ng-classeven': 'A',
        'ng-classodd': 'A',
        'ng-click': 'A',
        'ng-cloak': 'A',
        'ng-controller': 'A',
        'ng-copy': 'A',
        'ng-csp': 'A',
        'ng-cut': 'A',
        'ng-dblclick': 'A',
        'ng-disabled': 'A',
        'ng-dirty': 'A',
        'ng-focus': 'A',
        'ng-form': 'A',
        'ng-hide': 'A',
        'ng-hint': 'A',
        'ng-hint-exclude': 'A',
        'ng-hint-include': 'A',
        'ng-href': 'A',
        'ng-if': 'A',
        'ng-include': 'A',
        'ng-init': 'A',
        'ng-invalid': 'A',
        'ng-keydown': 'A',
        'ng-keypress': 'A',
        'ng-keyup': 'A',
        'ng-list': 'A',
        'ng-maxlength': 'A',
        'ng-minlength': 'A',
        'ng-model': 'A',
        'ng-model-options': 'A',
        'ng-mousedown': 'A',
        'ng-mouseenter': 'A',
        'ng-mouseleave': 'A',
        'ng-mousemove': 'A',
        'ng-mouseover': 'A',
        'ng-mouseup': 'A',
        'ng-nonbindable': 'A',
        'ng-open': 'A',
        'ng-options': 'A',
        'ng-paste': 'A',
        'ng-pattern': 'A',
        'ng-pluralize': 'A',
        'ng-pristine': 'A',
        'ng-readonly': 'A',
        'ng-repeat': 'A',
        'ng-repeat-start': 'A',
        'ng-repeat-end': 'A',
        'ng-required': 'A',
        'ng-selected': 'A',
        'ng-show': 'A',
        'ng-src': 'A',
        'ng-srcset': 'A',
        'ng-style': 'A',
        'ng-submit': 'A',
        'ng-switch': 'A',
        'ng-switch-when': 'A',
        'ng-transclude': 'A',
        'ng-true-value': 'A',
        'ng-trim': 'A',
        'ng-false-value': 'A',
        'ng-value': 'A',
        'ng-valid': 'A',
        'ng-view': 'A',
        'required': 'A',
        'when': 'A'
      }
    },
    'angular-custom-directives': {
      message: 'There was an AngularJS error in ',
      directives: {}
    },
    'angular-deprecated-directives': {
      message: 'There was an AngularJS error in ',
      directives: {
        'ng-bind-html-unsafe': 'deprecated'
      }
    }
  }
};

},{}],18:[function(require,module,exports){
var areSimilarEnough = require('./areSimilarEnough');
var levenshteinDistance = require('./levenshtein');

/**
 *@param directiveTypeData: {} with list of directives/attributes and
 *their respective restrict properties.
 *@param attribute: attribute name as string e.g. 'ng-click', 'width', 'src', etc.
 *
 *@return {} with Levenshtein Distance and name of the closest match to given attribute.
 **/
module.exports = function(directiveTypeData, attribute) {
  if(typeof attribute !== 'string') {
    throw new Error('Function must be passed a string as second parameter.');
  }
  if((directiveTypeData === null || directiveTypeData === undefined) ||
    typeof directiveTypeData !== 'object') {
    throw new Error('Function must be passed a defined object as first parameter.');
  }
  var min_levDist = Infinity,
      closestMatch = '';

  for(var directive in directiveTypeData){
    if(min_levDist !== 1 && areSimilarEnough(attribute,directive)) {
      var currentlevDist = levenshteinDistance(attribute, directive);
      closestMatch = (currentlevDist < min_levDist)? directive : closestMatch;
      min_levDist = (currentlevDist < min_levDist)? currentlevDist : min_levDist;
    }
  }
  return {min_levDist: min_levDist, match: closestMatch};
};

},{"./areSimilarEnough":4,"./levenshtein":30}],19:[function(require,module,exports){

var getFailedAttributesOfElement = require('./getFailedAttributesOfElement');

module.exports = function(scopeElements, options) {
  return scopeElements.map(getFailedAttributesOfElement.bind(null, options))
      .filter(function(x) {return x;});
};

},{"./getFailedAttributesOfElement":23}],20:[function(require,module,exports){
var ddLibData = require('./ddLib-data');

module.exports = function(dirName, attributes) {
  attributes = attributes.map(function(x){return x.nodeName;});
  var directive = ddLibData.directiveTypes['angular-custom-directives'].directives[dirName];
  var missing = [];
  if (directive && directive.require) {
    for (var i = 0, length = directive.require.length; i < length; i++) {
      var dirName = directive.require[i].directiveName;
      if (attributes.indexOf(dirName) < 0) {
        missing.push(dirName);
      }
    }
  }
  return missing;
};

},{"./ddLib-data":17}],21:[function(require,module,exports){
var hintLog = angular.hint = require('angular-hint-log'),
  MODULE_NAME = 'Directives';

var build = {
  deprecated: require('./buildDeprecated'),
  missingrequired: require('./buildMissingRequired'),
  mutuallyexclusive: require('./buildMutuallyExclusive'),
  ngevent: require('./buildNgEvent'),
  ngrepeatformat: require('./buildNgRepeatFormat'),
  nonexsisting: require('./buildNonExsisting'),
  wronguse: require('./buildWrongUse')
};

/**
 *@param failedElements: [] of {}s of all failed elements with their failed attributes and closest
 *matches or restrict properties
 *
 *@return [] of failed messages.
 **/
module.exports = function(failedElements) {
  failedElements.forEach(function(obj) {
    obj.data.forEach(function(info) {
      var id = (obj.domElement.id) ? ' with id: #' + obj.domElement.id : '',
        type = obj.domElement.nodeName,
        messageAndSeverity = build[info.typeError](info, id, type);
      hintLog.logMessage(MODULE_NAME, messageAndSeverity[0], messageAndSeverity[1]);
    });
  });
};

},{"./buildDeprecated":6,"./buildMissingRequired":7,"./buildMutuallyExclusive":8,"./buildNgEvent":10,"./buildNgRepeatFormat":11,"./buildNonExsisting":12,"./buildWrongUse":14,"angular-hint-log":47}],22:[function(require,module,exports){
var normalizeAttribute = require('./normalizeAttribute');
var ddLibData = require('./ddLib-data');
var isMutExclusiveDir = require('./isMutExclusiveDir');
var hasMutExclusivePair = require('./hasMutExclusivePair');
var attributeExsistsInTypes = require('./attributeExsistsInTypes');
var getSuggestions = require('./getSuggestions');
var checkNgRepeatFormat = require('./checkNgRepeatFormat');

/**
 *@param attributes: [] of attributes from element (includes tag name of element, e.g. DIV, P, etc.)
 *@param options: {} options object from beginSearch
 *
 *@return [] of failedAttributes with their respective suggestions and directiveTypes
 **/
module.exports = function(attributes, options) {
  var failedAttrs = [], mutExPairFound = false;
  for (var i = 0; i < attributes.length; i++) {
    var attr = normalizeAttribute(attributes[i].nodeName);
    var dirVal = ddLibData.directiveTypes['html-directives'].directives[attr] ||
      ddLibData.directiveTypes['angular-deprecated-directives'].directives[attr] || '';

    if(dirVal === 'deprecated') {
      failedAttrs.push({
        error: attr,
        directiveType: 'angular-deprecated-directives',
        typeError: 'deprecated'
      });
    }

    //if attr is a event attr. Html event directives are prefixed with ! in ddLibData
    if (dirVal.indexOf('!') > -1) {
      failedAttrs.push({
        error: attr,
        directiveType: 'html-directives',
        typeError: 'ngevent'
      });
      continue;
    }
    if (!mutExPairFound && isMutExclusiveDir(attr) && hasMutExclusivePair(attr, attributes)) {
      failedAttrs.push({
        error: attr,
        directiveType: 'angular-default-directives',
        typeError: 'mutuallyexclusive'
      });
      mutExPairFound = true;
      continue;
    }
    var attrVal = attributes[i].value || '';
    if(attr === 'ng-repeat') {
      var result = checkNgRepeatFormat(attrVal);
      if(result) {
        failedAttrs.push({
          error: attr,
          suggestion: result,
          directiveType: 'angular-default-directives',
          typeError: 'ngrepeatformat'
        });
      }
    }

    var result = attributeExsistsInTypes(attr,options);

    var suggestion = result.typeError === 'nonexsisting' ?
        getSuggestions(attr, options) : {match: ''};

    if (result.typeError) {
      failedAttrs.push({
        match: suggestion.match || '',
        wrongUse: result.wrongUse || '',
        error: attr,
        directiveType: suggestion.directiveType || 'angular-custom-directives',
        typeError: result.typeError
      });
    }
  }
  return failedAttrs;
};
},{"./attributeExsistsInTypes":5,"./checkNgRepeatFormat":15,"./ddLib-data":17,"./getSuggestions":25,"./hasMutExclusivePair":26,"./isMutExclusiveDir":29,"./normalizeAttribute":31}],23:[function(require,module,exports){
var getFailedAttributes = require('./getFailedAttributes');
var findMissingAttrs = require('./findMissingAttrs');


/**
 *@description
 *Adds element tag name (DIV, P, SPAN) to list of attributes with '*' prepended
 *for identification later.
 *
 *@param options: {} options object from beginSearch
 *@param element: HTML element to check attributes of
 *
 *@return {} of html element and [] of failed attributes
 **/
module.exports = function(options, element) {
  if(element.attributes.length) {
    var eleName = element.nodeName.toLowerCase();
    var eleAttrs = Array.prototype.slice.call(element.attributes);
    eleAttrs.push({
      nodeName: '*'+eleName
    });
    var failedAttrs = getFailedAttributes(eleAttrs, options);
    var missingRequired = findMissingAttrs(eleName, eleAttrs);
    var missingLength = missingRequired.length;

    if(failedAttrs.length || missingLength) {
      if(missingLength) {
        failedAttrs.push({
          directiveType: 'angular-custom-directive',
          missing: missingRequired,
          typeError: 'missingrequired'
        });
      }
      return {
        domElement: element,
        data: failedAttrs
      };
    }
  }
};

},{"./findMissingAttrs":20,"./getFailedAttributes":22}],24:[function(require,module,exports){
module.exports = function(str) {
  var customDirectives = [],
      pairs = [],
      SCOPE_REGEXP = /scope\s*:\s*{\s*[^}]*['"]\s*}/,
      PROPERTY_REGEXP = /\w+\s*:\s*['"][a-zA-Z=@&]+['"]/g,
      KEYVAL_REGEXP = /(\w+)\s*:\s*['"](.+)['"]/;
  var matchScope = str.replace(/\n/g,'').match(SCOPE_REGEXP);
  var propertiesMatch = matchScope ? matchScope[0].match(PROPERTY_REGEXP) : undefined;

  if (matchScope && propertiesMatch) {
    propertiesMatch.map(function(str){
      var temp = str.match(KEYVAL_REGEXP);
      pairs.push({key: temp[1], value: temp[2]});
    });
    pairs.forEach(function(pair){
      var name = (['=', '@', '&'].indexOf(pair.value) !== -1)? pair.key : pair.value.substring(1);
      customDirectives.push({directiveName: name , restrict:'A'});
    });
  }
  return customDirectives;
};

},{}],25:[function(require,module,exports){
var ddLibData = require('./ddLib-data');
var findClosestMatchIn = require('./findClosestMatchIn');

/**
 *@param attribute: attribute name as string e.g. 'ng-click', 'width', 'src', etc.
 *@param options: {} options object from beginSearch.
 *
 *@return {} with closest match to attribute and the directive type it corresponds to.
 **/
module.exports = function(attribute, options) {
  var min_levDist = Infinity,
      match = '',
      dirType = '';

  options.directiveTypes.forEach(function(directiveType) {
    var isTag = attribute.charAt(0) === '*';
    var isCustomDir = directiveType === 'angular-custom-directives';
    if (!isTag || (isTag && isCustomDir)) {
      var directiveTypeData = ddLibData.directiveTypes[directiveType].directives;
      var tempMatch = findClosestMatchIn(directiveTypeData, attribute);
      if (tempMatch.min_levDist < options.tolerance && tempMatch.min_levDist < min_levDist) {
        match = tempMatch.match;
        dirType = directiveType;
        min_levDist = tempMatch.min_levDist;
      }
    }
  });
  return {
    match: match,
    directiveType: dirType
  };
};

},{"./ddLib-data":17,"./findClosestMatchIn":18}],26:[function(require,module,exports){
var isMutExclusiveDir = require('./isMutExclusiveDir');

module.exports = function(attr, attributes) {
  var pair = isMutExclusiveDir(attr);

  return attributes.some(function(otherAttr) {
    return otherAttr.nodeName === pair;
  });
};

},{"./isMutExclusiveDir":29}],27:[function(require,module,exports){
var dasherize = require('dasherize');
var validate = require('validate-element-name');

module.exports = function(str) {
  var dashStr = dasherize(str);
  var validated = !validate(dashStr).message ? true : false;
  //Check for message definition because validate-element-name returns true for things starting
  //with ng-, polymer-, and data- but message is defined for those and errors.
  return validated && str.toLowerCase() !== str;
};

},{"dasherize":34,"validate-element-name":35}],28:[function(require,module,exports){
module.exports = function(facStr) {
  return facStr.match(/replace\s*:/);
};

},{}],29:[function(require,module,exports){
module.exports = function (dirName) {
  var exclusiveDirHash = {
    'ng-show' : 'ng-hide',
    'ng-hide' : 'ng-show',
    'ng-switch-when' : 'ng-switch-default',
    'ng-switch-default' : 'ng-switch-when',
  };
  return exclusiveDirHash[dirName];
};

},{}],30:[function(require,module,exports){
/**
 *@param s: first string to compare for Levenshtein Distance.
 *@param t: second string to compare for Levenshtein Distance.
 *
 *@description
 *Calculates the minimum number of changes (insertion, deletion, transposition) to get from s to t.
 *
 *credit: http://stackoverflow.com/questions/11919065/sort-an-array-by-the-levenshtein-distance-with-best-performance-in-javascript
 *http://www.merriampark.com/ld.htm, http://www.mgilleland.com/ld/ldjavascript.htm, Damerau–Levenshtein distance (Wikipedia)
 **/
module.exports = function(s, t) {
  if(typeof s !== 'string' || typeof t !== 'string') {
    throw new Error('Function must be passed two strings, given: '+typeof s+' and '+typeof t+'.');
  }
  var d = [];
  var n = s.length;
  var m = t.length;

  if (n === 0) {return m;}
  if (m === 0) {return n;}

  for (var ii = n; ii >= 0; ii--) { d[ii] = []; }
  for (var ii = n; ii >= 0; ii--) { d[ii][0] = ii; }
  for (var jj = m; jj >= 0; jj--) { d[0][jj] = jj; }
  for (var i = 1; i <= n; i++) {
    var s_i = s.charAt(i - 1);

    for (var j = 1; j <= m; j++) {
      if (i == j && d[i][j] > 4) return n;
      var t_j = t.charAt(j - 1);
      var cost = (s_i == t_j) ? 0 : 1;
      var mi = d[i - 1][j] + 1;
      var b = d[i][j - 1] + 1;
      var c = d[i - 1][j - 1] + cost;
      if (b < mi) mi = b;
      if (c < mi) mi = c;
      d[i][j] = mi;
      if (i > 1 && j > 1 && s_i == t.charAt(j - 2) && s.charAt(i - 2) == t_j) {
          d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + cost);
      }
    }
  }
  return d[n][m];
};

},{}],31:[function(require,module,exports){
/**
 *@param attribute: attribute name before normalization as string
 * e.g. 'data-ng-click', 'width', 'x:ng:src', etc.
 *
 *@return normalized attribute name
 **/
module.exports = function(attribute) {
  return attribute.replace(/^(?:data|x)[-_:]/,'').replace(/[:_]/g,'-');
};

},{}],32:[function(require,module,exports){

var formatResults = require('./formatResults');
var findFailedElements = require('./findFailedElements');
var setCustomDirectives = require('./setCustomDirectives');
var defaultTypes = [
  'html-directives',
  'angular-default-directives',
  'angular-custom-directives',
  'angular-deprecated-directives'
];


/**
 *
 *@param scopeElements: [] of HTML elements to be checked for incorrect attributes
 *@param customDirectives: [] of custom directive objects from $compile decorator
 *@param options: {} of options for app to run with:
 *    options.tolerance: Integer, maximum Levenshtein Distance to be allowed for misspellings
 *    options.directiveTypes: [] of which type of directives/attributes to search through
 **/
module.exports = function(scopeElements, customDirectives, options) {
  if(!Array.isArray(scopeElements)) {
    throw new Error('Function search must be passed an array.');
  }
  options = options || {};
  options.directiveTypes = options.directiveTypes || defaultTypes;
  options.tolerance = options.tolerance || 4;
  if(customDirectives && customDirectives.length){
    setCustomDirectives(customDirectives);
  }
  var failedElements = findFailedElements(scopeElements, options);
  formatResults(failedElements);
};

},{"./findFailedElements":19,"./formatResults":21,"./setCustomDirectives":33}],33:[function(require,module,exports){
var ddLibData = require('../lib/ddLib-data');

module.exports = function(customDirectives) {
  customDirectives.forEach(function(directive) {
    var directiveName = directive.directiveName.replace(/([A-Z])/g, '-$1').toLowerCase();
    ddLibData.directiveTypes['angular-custom-directives']
      .directives[directiveName] = directive;
  });
};

},{"../lib/ddLib-data":17}],34:[function(require,module,exports){
'use strict';

var isArray = Array.isArray || function (obj) {
  return Object.prototype.toString.call(obj) === '[object Array]';
};

var isDate = function (obj) {
  return Object.prototype.toString.call(obj) === '[object Date]';
};

var isRegex = function (obj) {
  return Object.prototype.toString.call(obj) === '[object RegExp]';
};

var has = Object.prototype.hasOwnProperty;
var objectKeys = Object.keys || function (obj) {
  var keys = [];
  for (var key in obj) {
    if (has.call(obj, key)) {
      keys.push(key);
    }
  }
  return keys;
};

function dashCase(str) {
  return str.replace(/([A-Z])/g, function ($1) {
    return '-' + $1.toLowerCase();
  });
}

function map(xs, f) {
  if (xs.map) {
    return xs.map(f);
  }
  var res = [];
  for (var i = 0; i < xs.length; i++) {
    res.push(f(xs[i], i));
  }
  return res;
}

function reduce(xs, f, acc) {
  if (xs.reduce) {
    return xs.reduce(f, acc);
  }
  for (var i = 0; i < xs.length; i++) {
    acc = f(acc, xs[i], i);
  }
  return acc;
}

function walk(obj) {
  if (!obj || typeof obj !== 'object') {
    return obj;
  }
  if (isDate(obj) || isRegex(obj)) {
    return obj;
  }
  if (isArray(obj)) {
    return map(obj, walk);
  }
  return reduce(objectKeys(obj), function (acc, key) {
    var camel = dashCase(key);
    acc[camel] = walk(obj[key]);
    return acc;
  }, {});
}

module.exports = function (obj) {
  if (typeof obj === 'string') {
    return dashCase(obj);
  }
  return walk(obj);
};

},{}],35:[function(require,module,exports){
'use strict';
var ncname = require('ncname');

var reservedNames = [
	'annotation-xml',
	'color-profile',
	'font-face',
	'font-face-src',
	'font-face-uri',
	'font-face-format',
	'font-face-name',
	'missing-glyph'
];

function hasError(name) {
	if (!name) {
		return 'Missing element name.';
	}

	if (/[A-Z]/.test(name)) {
		return 'Custom element names must not contain uppercase ASCII characters.';
	}

	if (name.indexOf('-') === -1) {
		return 'Custom element names must contain a hyphen. Example: unicorn-cake';
	}

	if (/^\d/i.test(name)) {
		return 'Custom element names must not start with a digit.';
	}

	if (/^-/i.test(name)) {
		return 'Custom element names must not start with a hyphen.';
	}

	// http://www.w3.org/TR/custom-elements/#concepts
	if (!ncname.test(name)) {
		return 'Invalid element name.';
	}

	if (reservedNames.indexOf(name) !== -1) {
		return 'The supplied element name is reserved and can\'t be used.\nSee: http://www.w3.org/TR/custom-elements/#concepts';
	}
}

function hasWarning(name) {
	if (/^polymer-/i.test(name)) {
		return 'Custom element names should not start with `polymer-`.\nSee: http://webcomponents.github.io/articles/how-should-i-name-my-element';
	}

	if (/^x-/i.test(name)) {
		return 'Custom element names should not start with `x-`.\nSee: http://webcomponents.github.io/articles/how-should-i-name-my-element/';
	}

	if (/^ng-/i.test(name)) {
		return 'Custom element names should not start with `ng-`.\nSee: http://docs.angularjs.org/guide/directive#creating-directives';
	}

	if (/^xml/i.test(name)) {
		return 'Custom element names should not start with `xml`.';
	}

	if (/^[^a-z]/i.test(name)) {
		return 'This element name is only valid in XHTML, not in HTML. First character should be in the range a-z.';
	}

	if (/[^a-z0-9]$/i.test(name)) {
		return 'Custom element names should not end with a non-alpha character.';
	}

	if (/[\.]/.test(name)) {
		return 'Custom element names should not contain a dot character as it would need to be escaped in a CSS selector.';
	}

	if (/[^\x20-\x7E]/.test(name)) {
		return 'Custom element names should not contain non-ASCII characters.';
	}

	if (/--/.test(name)) {
		return 'Custom element names should not contain consecutive hyphens.';
	}

	if (/[^a-z0-9]{2}/i.test(name)) {
		return 'Custom element names should not contain consecutive non-alpha characters.';
	}
}

module.exports = function (name) {
	var errMsg = hasError(name);

	return {
		isValid: !errMsg,
		message: errMsg || hasWarning(name)
	};
};

},{"ncname":36}],36:[function(require,module,exports){
'use strict';
var xmlChars = require('xml-char-classes');

function getRange(re) {
	return re.source.slice(1, -1);
}

// http://www.w3.org/TR/1999/REC-xml-names-19990114/#NT-NCName
module.exports = new RegExp('^[' + getRange(xmlChars.letter) + '_][' + getRange(xmlChars.letter) + getRange(xmlChars.digit) + '\\.\\-_' + getRange(xmlChars.combiningChar) + getRange(xmlChars.extender) + ']*$');

},{"xml-char-classes":37}],37:[function(require,module,exports){
exports.baseChar = /[A-Za-z\xC0-\xD6\xD8-\xF6\xF8-\u0131\u0134-\u013E\u0141-\u0148\u014A-\u017E\u0180-\u01C3\u01CD-\u01F0\u01F4\u01F5\u01FA-\u0217\u0250-\u02A8\u02BB-\u02C1\u0386\u0388-\u038A\u038C\u038E-\u03A1\u03A3-\u03CE\u03D0-\u03D6\u03DA\u03DC\u03DE\u03E0\u03E2-\u03F3\u0401-\u040C\u040E-\u044F\u0451-\u045C\u045E-\u0481\u0490-\u04C4\u04C7\u04C8\u04CB\u04CC\u04D0-\u04EB\u04EE-\u04F5\u04F8\u04F9\u0531-\u0556\u0559\u0561-\u0586\u05D0-\u05EA\u05F0-\u05F2\u0621-\u063A\u0641-\u064A\u0671-\u06B7\u06BA-\u06BE\u06C0-\u06CE\u06D0-\u06D3\u06D5\u06E5\u06E6\u0905-\u0939\u093D\u0958-\u0961\u0985-\u098C\u098F\u0990\u0993-\u09A8\u09AA-\u09B0\u09B2\u09B6-\u09B9\u09DC\u09DD\u09DF-\u09E1\u09F0\u09F1\u0A05-\u0A0A\u0A0F\u0A10\u0A13-\u0A28\u0A2A-\u0A30\u0A32\u0A33\u0A35\u0A36\u0A38\u0A39\u0A59-\u0A5C\u0A5E\u0A72-\u0A74\u0A85-\u0A8B\u0A8D\u0A8F-\u0A91\u0A93-\u0AA8\u0AAA-\u0AB0\u0AB2\u0AB3\u0AB5-\u0AB9\u0ABD\u0AE0\u0B05-\u0B0C\u0B0F\u0B10\u0B13-\u0B28\u0B2A-\u0B30\u0B32\u0B33\u0B36-\u0B39\u0B3D\u0B5C\u0B5D\u0B5F-\u0B61\u0B85-\u0B8A\u0B8E-\u0B90\u0B92-\u0B95\u0B99\u0B9A\u0B9C\u0B9E\u0B9F\u0BA3\u0BA4\u0BA8-\u0BAA\u0BAE-\u0BB5\u0BB7-\u0BB9\u0C05-\u0C0C\u0C0E-\u0C10\u0C12-\u0C28\u0C2A-\u0C33\u0C35-\u0C39\u0C60\u0C61\u0C85-\u0C8C\u0C8E-\u0C90\u0C92-\u0CA8\u0CAA-\u0CB3\u0CB5-\u0CB9\u0CDE\u0CE0\u0CE1\u0D05-\u0D0C\u0D0E-\u0D10\u0D12-\u0D28\u0D2A-\u0D39\u0D60\u0D61\u0E01-\u0E2E\u0E30\u0E32\u0E33\u0E40-\u0E45\u0E81\u0E82\u0E84\u0E87\u0E88\u0E8A\u0E8D\u0E94-\u0E97\u0E99-\u0E9F\u0EA1-\u0EA3\u0EA5\u0EA7\u0EAA\u0EAB\u0EAD\u0EAE\u0EB0\u0EB2\u0EB3\u0EBD\u0EC0-\u0EC4\u0F40-\u0F47\u0F49-\u0F69\u10A0-\u10C5\u10D0-\u10F6\u1100\u1102\u1103\u1105-\u1107\u1109\u110B\u110C\u110E-\u1112\u113C\u113E\u1140\u114C\u114E\u1150\u1154\u1155\u1159\u115F-\u1161\u1163\u1165\u1167\u1169\u116D\u116E\u1172\u1173\u1175\u119E\u11A8\u11AB\u11AE\u11AF\u11B7\u11B8\u11BA\u11BC-\u11C2\u11EB\u11F0\u11F9\u1E00-\u1E9B\u1EA0-\u1EF9\u1F00-\u1F15\u1F18-\u1F1D\u1F20-\u1F45\u1F48-\u1F4D\u1F50-\u1F57\u1F59\u1F5B\u1F5D\u1F5F-\u1F7D\u1F80-\u1FB4\u1FB6-\u1FBC\u1FBE\u1FC2-\u1FC4\u1FC6-\u1FCC\u1FD0-\u1FD3\u1FD6-\u1FDB\u1FE0-\u1FEC\u1FF2-\u1FF4\u1FF6-\u1FFC\u2126\u212A\u212B\u212E\u2180-\u2182\u3041-\u3094\u30A1-\u30FA\u3105-\u312C\uAC00-\uD7A3]/;

exports.ideographic = /[\u3007\u3021-\u3029\u4E00-\u9FA5]/;

exports.letter = /[A-Za-z\xC0-\xD6\xD8-\xF6\xF8-\u0131\u0134-\u013E\u0141-\u0148\u014A-\u017E\u0180-\u01C3\u01CD-\u01F0\u01F4\u01F5\u01FA-\u0217\u0250-\u02A8\u02BB-\u02C1\u0386\u0388-\u038A\u038C\u038E-\u03A1\u03A3-\u03CE\u03D0-\u03D6\u03DA\u03DC\u03DE\u03E0\u03E2-\u03F3\u0401-\u040C\u040E-\u044F\u0451-\u045C\u045E-\u0481\u0490-\u04C4\u04C7\u04C8\u04CB\u04CC\u04D0-\u04EB\u04EE-\u04F5\u04F8\u04F9\u0531-\u0556\u0559\u0561-\u0586\u05D0-\u05EA\u05F0-\u05F2\u0621-\u063A\u0641-\u064A\u0671-\u06B7\u06BA-\u06BE\u06C0-\u06CE\u06D0-\u06D3\u06D5\u06E5\u06E6\u0905-\u0939\u093D\u0958-\u0961\u0985-\u098C\u098F\u0990\u0993-\u09A8\u09AA-\u09B0\u09B2\u09B6-\u09B9\u09DC\u09DD\u09DF-\u09E1\u09F0\u09F1\u0A05-\u0A0A\u0A0F\u0A10\u0A13-\u0A28\u0A2A-\u0A30\u0A32\u0A33\u0A35\u0A36\u0A38\u0A39\u0A59-\u0A5C\u0A5E\u0A72-\u0A74\u0A85-\u0A8B\u0A8D\u0A8F-\u0A91\u0A93-\u0AA8\u0AAA-\u0AB0\u0AB2\u0AB3\u0AB5-\u0AB9\u0ABD\u0AE0\u0B05-\u0B0C\u0B0F\u0B10\u0B13-\u0B28\u0B2A-\u0B30\u0B32\u0B33\u0B36-\u0B39\u0B3D\u0B5C\u0B5D\u0B5F-\u0B61\u0B85-\u0B8A\u0B8E-\u0B90\u0B92-\u0B95\u0B99\u0B9A\u0B9C\u0B9E\u0B9F\u0BA3\u0BA4\u0BA8-\u0BAA\u0BAE-\u0BB5\u0BB7-\u0BB9\u0C05-\u0C0C\u0C0E-\u0C10\u0C12-\u0C28\u0C2A-\u0C33\u0C35-\u0C39\u0C60\u0C61\u0C85-\u0C8C\u0C8E-\u0C90\u0C92-\u0CA8\u0CAA-\u0CB3\u0CB5-\u0CB9\u0CDE\u0CE0\u0CE1\u0D05-\u0D0C\u0D0E-\u0D10\u0D12-\u0D28\u0D2A-\u0D39\u0D60\u0D61\u0E01-\u0E2E\u0E30\u0E32\u0E33\u0E40-\u0E45\u0E81\u0E82\u0E84\u0E87\u0E88\u0E8A\u0E8D\u0E94-\u0E97\u0E99-\u0E9F\u0EA1-\u0EA3\u0EA5\u0EA7\u0EAA\u0EAB\u0EAD\u0EAE\u0EB0\u0EB2\u0EB3\u0EBD\u0EC0-\u0EC4\u0F40-\u0F47\u0F49-\u0F69\u10A0-\u10C5\u10D0-\u10F6\u1100\u1102\u1103\u1105-\u1107\u1109\u110B\u110C\u110E-\u1112\u113C\u113E\u1140\u114C\u114E\u1150\u1154\u1155\u1159\u115F-\u1161\u1163\u1165\u1167\u1169\u116D\u116E\u1172\u1173\u1175\u119E\u11A8\u11AB\u11AE\u11AF\u11B7\u11B8\u11BA\u11BC-\u11C2\u11EB\u11F0\u11F9\u1E00-\u1E9B\u1EA0-\u1EF9\u1F00-\u1F15\u1F18-\u1F1D\u1F20-\u1F45\u1F48-\u1F4D\u1F50-\u1F57\u1F59\u1F5B\u1F5D\u1F5F-\u1F7D\u1F80-\u1FB4\u1FB6-\u1FBC\u1FBE\u1FC2-\u1FC4\u1FC6-\u1FCC\u1FD0-\u1FD3\u1FD6-\u1FDB\u1FE0-\u1FEC\u1FF2-\u1FF4\u1FF6-\u1FFC\u2126\u212A\u212B\u212E\u2180-\u2182\u3007\u3021-\u3029\u3041-\u3094\u30A1-\u30FA\u3105-\u312C\u4E00-\u9FA5\uAC00-\uD7A3]/;

exports.combiningChar = /[\u0300-\u0345\u0360\u0361\u0483-\u0486\u0591-\u05A1\u05A3-\u05B9\u05BB-\u05BD\u05BF\u05C1\u05C2\u05C4\u064B-\u0652\u0670\u06D6-\u06E4\u06E7\u06E8\u06EA-\u06ED\u0901-\u0903\u093C\u093E-\u094D\u0951-\u0954\u0962\u0963\u0981-\u0983\u09BC\u09BE-\u09C4\u09C7\u09C8\u09CB-\u09CD\u09D7\u09E2\u09E3\u0A02\u0A3C\u0A3E-\u0A42\u0A47\u0A48\u0A4B-\u0A4D\u0A70\u0A71\u0A81-\u0A83\u0ABC\u0ABE-\u0AC5\u0AC7-\u0AC9\u0ACB-\u0ACD\u0B01-\u0B03\u0B3C\u0B3E-\u0B43\u0B47\u0B48\u0B4B-\u0B4D\u0B56\u0B57\u0B82\u0B83\u0BBE-\u0BC2\u0BC6-\u0BC8\u0BCA-\u0BCD\u0BD7\u0C01-\u0C03\u0C3E-\u0C44\u0C46-\u0C48\u0C4A-\u0C4D\u0C55\u0C56\u0C82\u0C83\u0CBE-\u0CC4\u0CC6-\u0CC8\u0CCA-\u0CCD\u0CD5\u0CD6\u0D02\u0D03\u0D3E-\u0D43\u0D46-\u0D48\u0D4A-\u0D4D\u0D57\u0E31\u0E34-\u0E3A\u0E47-\u0E4E\u0EB1\u0EB4-\u0EB9\u0EBB\u0EBC\u0EC8-\u0ECD\u0F18\u0F19\u0F35\u0F37\u0F39\u0F3E\u0F3F\u0F71-\u0F84\u0F86-\u0F8B\u0F90-\u0F95\u0F97\u0F99-\u0FAD\u0FB1-\u0FB7\u0FB9\u20D0-\u20DC\u20E1\u302A-\u302F\u3099\u309A]/;

exports.digit = /[0-9\u0660-\u0669\u06F0-\u06F9\u0966-\u096F\u09E6-\u09EF\u0A66-\u0A6F\u0AE6-\u0AEF\u0B66-\u0B6F\u0BE7-\u0BEF\u0C66-\u0C6F\u0CE6-\u0CEF\u0D66-\u0D6F\u0E50-\u0E59\u0ED0-\u0ED9\u0F20-\u0F29]/;

exports.extender = /[\xB7\u02D0\u02D1\u0387\u0640\u0E46\u0EC6\u3005\u3031-\u3035\u309D\u309E\u30FC-\u30FE]/;
},{}],38:[function(require,module,exports){
'use strict';

/**
* Load necessary functions from /lib into variables.
*/
var ngEventDirectives = require('./lib/getEventDirectives')(),
  getEventAttribute = require('./lib/getEventAttribute'),
  getFunctionNames = require('./lib/getFunctionNames'),
  formatResults = require('./lib/formatResults');

/**
* Decorate $provide in order to examine ng-event directives
* and hint about their effective use.
*/
angular.module('ngHintEvents', [])
  .config(['$provide', function($provide) {

    for(var directive in ngEventDirectives) {
      var dirName = ngEventDirectives[directive]+'Directive';
      try{
        $provide.decorator(dirName, ['$delegate', '$timeout', '$parse',
          function($delegate, $timeout, $parse) {

            var original = $delegate[0].compile, falseBinds = [], messages = [];

            $delegate[0].compile = function(element, attrs, transclude) {
              var angularAttrs = attrs.$attr;
              var eventAttrName = getEventAttribute(angularAttrs);
              var fn = $parse(attrs[eventAttrName]);
              var messages = [];
              return function ngEventHandler(scope, element, attrs) {
                for(var attr in angularAttrs) {
                  var boundFuncs = getFunctionNames(attrs[attr]);
                  boundFuncs.forEach(function(boundFn) {
                    if(ngEventDirectives[attr] && !(boundFn in scope)) {
                      messages.push({
                        scope: scope,
                        element:element,
                        attrs: attrs,
                        boundFunc: boundFn
                      });
                    }
                  });
                }
                element.on(eventAttrName.substring(2).toLowerCase(), function(event) {
                  scope.$apply(function() {
                    fn(scope, {$event:event});
                  });
                });
                formatResults(messages);
              };
            };
            return $delegate;
        }]);
      } catch(e) {}
    }
  }]);
},{"./lib/formatResults":40,"./lib/getEventAttribute":41,"./lib/getEventDirectives":42,"./lib/getFunctionNames":43}],39:[function(require,module,exports){
'use strict';

var getValidProps = require('./getValidProps'),
  suggest = require('suggest-it');

module.exports = function addSuggestions(messages) {
  messages.forEach(function(messageObj) {
    var dictionary = getValidProps(messageObj.scope),
      suggestion = suggest(dictionary)(messageObj.boundFunc);
    messageObj['match'] = suggestion;
  });
  return messages;
};

},{"./getValidProps":44,"suggest-it":46}],40:[function(require,module,exports){
'use strict';

var hintLog = angular.hint = require('angular-hint-log'),
  addSuggestions = require('./addSuggestions'),
  MODULE_NAME = 'Events',
  SEVERITY_ERROR = 1;

module.exports = function formatResults(messages) {
  messages = addSuggestions(messages);
  if(messages.length) {
    messages.forEach(function(obj) {
      var id = (obj.element[0].id) ? ' with id: #' + obj.element[0].id : '',
        type = obj.element[0].nodeName,
        suggestion = obj.match ? ' (Try "' + obj.match + '").': '.',
        message = 'Variable "' + obj.boundFunc + '" called on ' + type + ' element' + id +
          ' does not exist in that scope' + suggestion + ' Event directive found on "' +
          obj.element[0].outerHTML + '".';
      hintLog.logMessage(MODULE_NAME, message, SEVERITY_ERROR);
    });
  }
};

},{"./addSuggestions":39,"angular-hint-log":47}],41:[function(require,module,exports){
'use strict';

var ngEventDirectives = require('./getEventDirectives')();

module.exports = function getEventAttribute(attrs) {
  for(var attr in attrs) {
    if(ngEventDirectives[attr]) {
      return attr;
    }
  }
};

},{"./getEventDirectives":42}],42:[function(require,module,exports){
'use strict';

module.exports = function getEventDirectives() {
  var list = 'click submit mouseenter mouseleave mousemove mousedown mouseover mouseup dblclick keyup keydown keypress blur focus submit cut copy paste'.split(' ');
  var eventDirHash = {};
  list.forEach(function(dirName) {
    dirName = 'ng'+dirName.charAt(0).toUpperCase()+dirName.substring(1);
    eventDirHash[dirName] = dirName;
  });
  return eventDirHash;
};

},{}],43:[function(require,module,exports){
'use strict';

module.exports = function getFunctionNames(str) {
  var results = str.replace(/\s+/g, '').split(/[\+\-\/\|\<\>\^=&!%~]/g).map(function(x) {
    if(isNaN(+x)) {
      if(x.match(/\w+\(.*\)$/)){
        return x.substring(0, x.indexOf('('));
      }
      return x;
    }
  }).filter(function(x){return x;});
  return results;
};

},{}],44:[function(require,module,exports){
'use strict';

module.exports = function getValidProps(obj) {
  var props = [];
  for(var prop in obj) {
    if (prop.charAt(0) !== '$' && typeof obj[prop] === 'function') {
      props.push(prop);
    }
  }
  return props;
};

},{}],45:[function(require,module,exports){
module.exports = distance;

function distance(a, b) {
  var table = [];
  if (a.length === 0 || b.length === 0) return Math.max(a.length, b.length);
  for (var ii = 0, ilen = a.length + 1; ii !== ilen; ++ii) {
    table[ii] = [];
    for (var jj = 0, jlen = b.length + 1; jj !== jlen; ++jj) {
      if (ii === 0 || jj === 0) table[ii][jj] = Math.max(ii, jj);
      else {
        var diagPenalty = Number(a[ii-1] !== b[jj-1]);
        var diag = table[ii - 1][jj - 1] + diagPenalty;
        var top = table[ii - 1][jj] + 1;
        var left = table[ii][jj - 1] + 1;
        table[ii][jj] = Math.min(left, top, diag);
      }
    }
  }
  return table[a.length][b.length];
}


},{}],46:[function(require,module,exports){
module.exports = suggestDictionary;

var distance = require('./levenstein_distance');

function suggestDictionary(dict, opts) {
  opts = opts || {};
  var threshold = opts.threshold || 0.5;
  return function suggest(word) {
    var length = word.length;
    return dict.reduce(function (result, dictEntry) {
      var score = distance(dictEntry, word);
      if (result.score > score && score / length < threshold) {
        result.score = score;
        result.word = dictEntry;
      }
      return result;
    }, { score: Infinity }).word;
  };
}

suggestDictionary.distance = distance;

},{"./levenstein_distance":45}],47:[function(require,module,exports){
/**
* HintLog creates a queue of messages logged by ngHint modules. This object
* has a key for each ngHint module that corresponds to the messages
* from that module.
*/
var queuedMessages = {},
    MESSAGE_TYPES = [
      'error',
      'warning',
      'suggestion'
    ];

/**
* Add a message to the HintLog message queue. Messages are organized into categories
* according to their module name and severity.
**/
function logMessage(moduleName, message, severity, category) {
  // If no severity was provided, categorize the message as a `suggestion`
  severity = severity || 3;
  var messageType = MESSAGE_TYPES[severity - 1];

  // If no ModuleName was found, categorize the message under `General`
  moduleName = moduleName || 'General';

  // If the category does not exist, initialize a new object
  queuedMessages[moduleName] = queuedMessages[moduleName] || {};
  queuedMessages[moduleName][messageType] = queuedMessages[moduleName][messageType] || [];

  if (queuedMessages[moduleName][messageType].indexOf(message) < 0) {
    queuedMessages[moduleName][messageType].push(message);
  }

  module.exports.onMessage(moduleName, message, messageType, category);
}

/**
* Return and empty the current queue of messages.
**/
function flush() {
  var flushMessages = queuedMessages;
  queuedMessages = {};
  return flushMessages;
}

module.exports.onMessage = function(message) {};
module.exports.logMessage = logMessage;
module.exports.flush = flush;

},{}],48:[function(require,module,exports){
'use strict';

var storeDependencies = require('./lib/storeDependencies'),
  getModule = require('./lib/getModule'),
  start = require('./lib/start'),
  storeNgAppAndView = require('./lib/storeNgAppAndView'),
  storeUsedModules = require('./lib/storeUsedModules'),
  hasNameSpace = require('./lib/hasNameSpace'),
  modData = require('./lib/moduleData');

var doc = Array.prototype.slice.call(document.getElementsByTagName('*')),
  originalAngularModule = angular.module,
  modules = {};

storeNgAppAndView(doc);

angular.module = function() {
  var requiresOriginal = arguments[1],
    module = originalAngularModule.apply(this, arguments),
    name = module.name;
  module.requiresOriginal = requiresOriginal;
  modules[name] = module;
  hasNameSpace(name);
  var modToCheck = getModule(name, true);

  if(modToCheck && modToCheck.requiresOriginal !== module.requiresOriginal) {
    if(!modData.createdMulti[name]) {
      modData.createdMulti[name] = [getModule(name,true)];
    }
    modData.createdMulti[name].push(module);
  }
  modData.createdModules[name] = module;
  return module;
};

angular.module('ngHintModules', []).config(function() {
  var ngAppMod = modules[modData.ngAppMod];
  storeUsedModules(ngAppMod, modules);
  start();
});

},{"./lib/getModule":51,"./lib/hasNameSpace":55,"./lib/moduleData":57,"./lib/start":60,"./lib/storeDependencies":61,"./lib/storeNgAppAndView":62,"./lib/storeUsedModules":63}],49:[function(require,module,exports){
var hintLog = angular.hint = require('angular-hint-log'),
  MODULE_NAME = 'Modules';

module.exports = function(modules) {
  modules.forEach(function(module) {
    hintLog.logMessage(MODULE_NAME, module.message, module.severity);
  });
};

},{"angular-hint-log":47}],50:[function(require,module,exports){
var modData = require('./moduleData');
  MODULE_NAME = 'Modules',
  SEVERITY_WARNING = 2;

module.exports = function() {
  var multiLoaded = [];
  for(var modName in modData.createdMulti) {
    var message = 'Multiple modules with name "' + modName + '" are being created and they will ' +
      'overwrite each other.';
    var multi = modData.createdMulti[modName];
    var multiLength = multi.length;
    var details = {
      existingModule: multi[multiLength - 1],
      overwrittenModules: multi.slice(0, multiLength - 1)
    };
    multiLoaded
      .push({module: details, message: message, name: MODULE_NAME, severity: SEVERITY_WARNING});
  }
  return multiLoaded;
};

},{"./moduleData":57}],51:[function(require,module,exports){
var modData = require('./moduleData');

module.exports = function(moduleName, getCreated) {
  return (getCreated)? modData.createdModules[moduleName] : modData.loadedModules[moduleName];
};

},{"./moduleData":57}],52:[function(require,module,exports){
var hintLog = angular.hint = require('angular-hint-log'),
  MODULE_NAME = 'Modules',
  SEVERITY_ERROR = 1;
 module.exports = function(attrs, ngAppFound) {
   if(attrs['ng-app'] && ngAppFound) {
     hintLog.logMessage(MODULE_NAME, 'ng-app may only be included once. The module "' +
      attrs['ng-app'].value + '" was not used to bootstrap because ng-app was already included.',
      SEVERITY_ERROR);
   }
  return attrs['ng-app'] ? attrs['ng-app'].value : undefined;
 };



},{"angular-hint-log":47}],53:[function(require,module,exports){
var getModule = require('./getModule'),
  dictionary = Object.keys(require('./moduleData').createdModules),
  suggest = require('suggest-it')(dictionary),
  SEVERITY_ERROR = 1;

module.exports = function(loadedModules) {
  var undeclaredModules = [];
  for(var module in loadedModules) {
    var cModule = getModule(module, true);
    if(!cModule) {
      var match = suggest(module),
        suggestion = (match) ? '; Try: "'+match+'"' : '',
        message = 'Module "'+module+'" was loaded but does not exist'+suggestion+'.';

      undeclaredModules.push({module: null, message: message, severity: SEVERITY_ERROR});
    }
  }
  return undeclaredModules;
};

},{"./getModule":51,"./moduleData":57,"suggest-it":65}],54:[function(require,module,exports){
var getModule = require('./getModule'),
  IGNORED = ['ngHintControllers', 'ngHintDirectives', 'ngHintDOM', 'ngHintEvents',
    'ngHintInterpolation', 'ngHintModules'];
  SEVERITY_WARNING = 2;

module.exports = function(createdModules) {
  var unusedModules = [];
  for(var module in createdModules) {
    if(!getModule(module)) {
      var cModule = createdModules[module],
        message = 'Module "' + cModule.name + '" was created but never loaded.';
      if(IGNORED.indexOf(cModule.name) === -1) {
        unusedModules.push({module: cModule, message: message, severity: SEVERITY_WARNING});
      }
    }
  }
  return unusedModules;
};

},{"./getModule":51}],55:[function(require,module,exports){
var hintLog = angular.hint = require('angular-hint-log'),
  MODULE_NAME = 'Modules',
  SEVERITY_SUGGESTION = 3;
module.exports = function(str) {
  if(str.toLowerCase() === str || str.charAt(0).toUpperCase() === str.charAt(0)) {
    hintLog.logMessage(MODULE_NAME, 'The best practice for' +
      ' module names is to use lowerCamelCase. Check the name of "' + str + '".',
      SEVERITY_SUGGESTION);
    return false;
  }
  return true;
};

},{"angular-hint-log":47}],56:[function(require,module,exports){
var normalizeAttribute = require('./normalizeAttribute');

module.exports = function(attrs) {
  for(var i = 0, length = attrs.length; i < length; i++) {
    if(normalizeAttribute(attrs[i].nodeName) === 'ng-view' ||
        attrs[i].value.indexOf('ng-view') > -1) {
          return true;
    }
  }
};

},{"./normalizeAttribute":59}],57:[function(require,module,exports){
module.exports = {
  createdModules: {},
  createdMulti: {},
  loadedModules: {}
};

},{}],58:[function(require,module,exports){
var modData = require('./moduleData'),
  getModule = require('./getModule');

module.exports = function() {
  if(modData.ngViewExists && !getModule('ngRoute')) {
    return {message: 'Directive "ngView" was used in the application however "ngRoute" was not loaded into any module.'};
  }
};

},{"./getModule":51,"./moduleData":57}],59:[function(require,module,exports){
module.exports = function(attribute) {
  return attribute.replace(/^(?:data|x)[-_:]/, '').replace(/[:_]/g, '-');
};

},{}],60:[function(require,module,exports){
var display = require('./display'),
  formatMultiLoaded = require('./formatMultiLoaded'),
  getUnusedModules = require('./getUnusedModules'),
  getUndeclaredModules = require('./getUndeclaredModules'),
  modData = require('./moduleData'),
  ngViewNoNgRoute = require('./ngViewNoNgRoute');

module.exports = function() {
  var unusedModules = getUnusedModules(modData.createdModules),
    undeclaredModules = getUndeclaredModules(modData.loadedModules),
    multiLoaded = formatMultiLoaded(),
    noNgRoute = ngViewNoNgRoute();
  if(unusedModules.length || undeclaredModules.length || multiLoaded.length || noNgRoute) {
    var toSend = unusedModules.concat(undeclaredModules)
      .concat(multiLoaded);
    if(noNgRoute) {
      toSend = toSend.concat(noNgRoute);
    }
    display(toSend);
  }
};

},{"./display":49,"./formatMultiLoaded":50,"./getUndeclaredModules":53,"./getUnusedModules":54,"./moduleData":57,"./ngViewNoNgRoute":58}],61:[function(require,module,exports){
var modData = require('./moduleData');

module.exports = function(module, isNgAppMod) {
  var name = module.name || module;
  if(!isNgAppMod){
    module.requires.forEach(function(dependency){
      modData.loadedModules[dependency] = dependency;
    });
  }
  else {
    modData.loadedModules[name] = name;
    modData.ngAppMod = name;
  }
};

},{"./moduleData":57}],62:[function(require,module,exports){
var getNgAppMod = require('./getNgAppMod'),
  inAttrsOrClasses = require('./inAttrsOrClasses'),
  storeDependencies = require('./storeDependencies'),
  modData = require('./moduleData');

module.exports = function(doms) {
  var bothFound,
      ngViewFound,
      elem,
      isElemName,
      isInAttrsOrClasses,
      ngAppMod;

  for(var i = 0; i < doms.length; i++) {
    elem = doms[i];
    var attributes = elem.attributes;
    isElemName = elem.nodeName.toLowerCase() === 'ng-view';
    isInAttrsOrClasses = inAttrsOrClasses(attributes);

    ngViewFound = isElemName || isInAttrsOrClasses;

    ngAppMod = getNgAppMod(attributes, modData.ngAppFound);
    modData.ngAppFound = modData.ngAppFound || ngAppMod;

    if(ngAppMod) {
      storeDependencies(ngAppMod, true);
    }
    modData.ngViewExists = ngViewFound ? true : modData.ngViewExists;

    if(bothFound) {
      break;
    }
  }
};

},{"./getNgAppMod":52,"./inAttrsOrClasses":56,"./moduleData":57,"./storeDependencies":61}],63:[function(require,module,exports){
var storeDependencies = require('./storeDependencies');

var storeUsedModules = module.exports = function(module, modules){
  if(module) {
    storeDependencies(module);
    module.requires.forEach(function(modName) {
      var mod = modules[modName];
      storeUsedModules(mod, modules);
    });
  }
};
},{"./storeDependencies":61}],64:[function(require,module,exports){
module.exports=require(45)
},{"/usr/local/google/home/sjelin/angular-hint/node_modules/angular-hint-events/node_modules/suggest-it/lib/levenstein_distance.js":45}],65:[function(require,module,exports){
module.exports=require(46)
},{"./levenstein_distance":64,"/usr/local/google/home/sjelin/angular-hint/node_modules/angular-hint-events/node_modules/suggest-it/lib/suggest-it.js":46}],66:[function(require,module,exports){
'use strict';

var summarize = require('./lib/summarize-model');
var hint = angular.hint = require('angular-hint-log');
var debounceOn = require('debounce-on');

hint.emit = function () {};

module.exports = angular.module('ngHintScopes', []).config(['$provide', function ($provide) {
  $provide.decorator('$rootScope', ['$delegate', '$parse', decorateRootScope]);
  $provide.decorator('$compile', ['$delegate', decorateDollaCompile]);
}]);

function decorateRootScope($delegate, $parse) {

  var perf = window.performance || { now: function () { return 0; } };

  var scopes = {},
      watching = {};

  var debouncedEmitModelChange = debounceOn(emitModelChange, 10, byScopeId);

  hint.watch = function (scopeId, path) {
    path = typeof path === 'string' ? path.split('.') : path;

    if (!watching[scopeId]) {
      watching[scopeId] = {};
    }

    for (var i = 1, ii = path.length; i <= ii; i += 1) {
      var partialPath = path.slice(0, i).join('.');
      if (watching[scopeId][partialPath]) {
        continue;
      }
      var get = gettterer(scopeId, partialPath);
      var value = summarize(get());
      watching[scopeId][partialPath] = {
        get: get,
        value: value
      };
      hint.emit('model:change', {
        id: scopeId,
        path: partialPath,
        value: value
      });
    }
  };

  hint.unwatch = function (scopeId, unwatchPath) {
    Object.keys(watching[scopeId]).
      forEach(function (path) {
        if (path.indexOf(unwatchPath) === 0) {
          delete watching[scopeId][path];
        }
      });
  };

  var debouncedEmit = debounceOn(hint.emit, 10, function (params) {
    return params.id + params.path;
  });


  var scopePrototype = ('getPrototypeOf' in Object) ?
      Object.getPrototypeOf($delegate) : $delegate.__proto__;

  var _watch = scopePrototype.$watch;
  scopePrototype.$watch = function (watchExpression, reactionFunction) {
    var watchStr = humanReadableWatchExpression(watchExpression);
    var scopeId = this.$id;
    if (typeof watchExpression === 'function') {
      arguments[0] = function () {
        var start = perf.now();
        var ret = watchExpression.apply(this, arguments);
        var end = perf.now();
        hint.emit('scope:watch', {
          id: scopeId,
          watch: watchStr,
          time: end - start
        });
        return ret;
      };
    } else {
      var thatScope = this;
      arguments[0] = function () {
        var start = perf.now();
        var ret = thatScope.$eval(watchExpression);
        var end = perf.now();
        hint.emit('scope:watch', {
          id: scopeId,
          watch: watchStr,
          time: end - start
        });
        return ret;
      };
    }

    if (typeof reactionFunction === 'function') {
      var applyStr = reactionFunction.toString();
      arguments[1] = function () {
        var start = perf.now();
        var ret = reactionFunction.apply(this, arguments);
        var end = perf.now();
        hint.emit('scope:reaction', {
          id: this.$id,
          watch: watchStr,
          time: end - start
        });
        return ret;
      };
    }

    return _watch.apply(this, arguments);
  };


  var _destroy = scopePrototype.$destroy;
  scopePrototype.$destroy = function () {
    var id = this.id;

    hint.emit('scope:destroy', { id: id });

    delete scopes[id];
    delete watching[id];

    return _destroy.apply(this, arguments);
  };


  var _new = scopePrototype.$new;
  scopePrototype.$new = function () {
    var child = _new.apply(this, arguments);

    scopes[child.$id] = child;
    watching[child.$id] = {};

    hint.emit('scope:new', { parent: this.$id, child: child.$id });
    setTimeout(function () {
      emitScopeElt(child);
    }, 0);
    return child;
  };

  function emitScopeElt (scope) {
    var scopeId = scope.$id;
    var elt = findElt(scopeId);
    var descriptor = scopeDescriptor(elt, scope);
    hint.emit('scope:link', {
      id: scopeId,
      descriptor: descriptor
    });
  }

  function findElt (scopeId) {
    var elts = document.querySelectorAll('.ng-scope');
    var elt, scope;

    for (var i = 0; i < elts.length; i++) {
      elt = angular.element(elts[i]);
      scope = elt.scope();
      if (scope.$id === scopeId) {
        return elt;
      }
    }
  }


  var _digest = scopePrototype.$digest;
  scopePrototype.$digest = function (fn) {
    var start = perf.now();
    var ret = _digest.apply(this, arguments);
    var end = perf.now();
    hint.emit('scope:digest', { id: this.$id, time: end - start });
    return ret;
  };


  var _apply = scopePrototype.$apply;
  scopePrototype.$apply = function (fn) {
    var start = perf.now();
    var ret = _apply.apply(this, arguments);
    var end = perf.now();
    hint.emit('scope:apply', { id: this.$id, time: end - start });
    debouncedEmitModelChange(this);
    return ret;
  };


  function gettterer (scopeId, path) {
    if (path === '') {
      return function () {
        return scopes[scopeId];
      };
    }
    var getter = $parse(path);
    return function () {
      return getter(scopes[scopeId]);
    };
  }

  function emitModelChange (scope) {
    var scopeId = scope.$id;
    if (watching[scopeId]) {
      Object.keys(watching[scopeId]).forEach(function (path) {
        var model = watching[scopeId][path];
        var value = summarize(model.get());
        if (value !== model.value) {
          hint.emit('model:change', {
            id: scope.$id,
            path: path,
            oldValue: model.value,
            value: value
          });
          model.value = value;
        }
      });
    }
  }

  hint.emit('scope:new', {
    parent: null,
    child: $delegate.$id
  });
  scopes[$delegate.$id] = $delegate;
  watching[$delegate.$id] = {};

  return $delegate;
}

function decorateDollaCompile ($delegate) {
  return function () {
    var link = $delegate.apply(this, arguments);

    return function (scope) {
      var elt = link.apply(this, arguments);
      var descriptor = scopeDescriptor(elt, scope);
      hint.emit('scope:link', {
        id: scope.$id,
        descriptor: descriptor
      });
      return elt;
    }
  }
}

function scopeDescriptor (elt, scope) {
  var val,
      types = [
        'ng-app',
        'ng-controller',
        'ng-repeat',
        'ng-include'
      ],
      theseTypes = [],
      type;

  if (elt) {
    for (var i = 0; i < types.length; i++) {
      type = types[i];
      if (val = elt.attr(type)) {
        theseTypes.push(type + '="' + val + '"');
      }
    }
  }
  if (theseTypes.length === 0) {
    return 'scope.$id=' + scope.$id;
  } else {
    return theseTypes.join(' ');
  }
}

function byScopeId (scope) {
  return scope.$id;
}

function humanReadableWatchExpression (fn) {
  if (fn.exp) {
    fn = fn.exp;
  } else if (fn.name) {
    fn = fn.name;
  }
  return fn.toString();
}

},{"./lib/summarize-model":67,"angular-hint-log":47,"debounce-on":68}],67:[function(require,module,exports){

module.exports = function summarizeModel (model) {

  if (model instanceof Array) {
    return JSON.stringify(model.map(summarizeProperty));
  } else if (typeof model === 'object') {
    return JSON.stringify(Object.
        keys(model).
        filter(isAngularPrivatePropertyName).
        reduce(shallowSummary, {}));
  } else {
    return model;
  }

  function shallowSummary (obj, prop) {
    obj[prop] = summarizeProperty(model[prop]);
    return obj;
  }
};

function isAngularPrivatePropertyName (key) {
  return !(key[0] === '$' && key[1] === '$') && key !== '$parent' && key !== '$root';
}

// TODO: handle DOM nodes, fns, etc better.
function summarizeProperty (obj) {
  return obj instanceof Array ?
      { '~array-length': obj.length } :
    obj === null ?
      null :
    typeof obj === 'object' ?
      { '~object': true } :
      obj;
}

},{}],68:[function(require,module,exports){
module.exports = function debounceOn (fn, timeout, hash) {
  var timeouts = {};

  timeout = typeof timeout === 'number' ? timeout : (hash = timeout, 100);
  hash = typeof hash === 'function' ? hash : defaultHash;

  return function () {
    var key = hash.apply(null, arguments);
    var args = arguments;
    if (typeof timeouts[key] === 'undefined') {
      timeouts[key] = setTimeout(function () {
        delete timeouts[key];
        fn.apply(null, args);
      }, timeout);
    }
    return function cancel () {
      if (timeouts[key]) {
        clearTimeout(timeouts[key]);
        delete timeouts[key];
        return true;
      }
      return false;
    };
  };
};

function defaultHash () {
  return Array.prototype.join.call(arguments, '::');
}

},{}]},{},[1]);

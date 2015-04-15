/*!
 * Angular Material Design
 * https://github.com/angular/material
 * @license MIT
 * v0.8.1
 */
goog.provide('ng.material.components.select');
goog.require('ng.material.components.backdrop');
goog.require('ng.material.core');
(function() {
'use strict';
/**
 * @ngdoc module
 * @name material.components.select
 */

/***************************************************

### TODO ###
**DOCUMENTATION AND DEMOS**

- [ ] ng-model with child mdOptions (basic)
- [ ] ng-model="foo" ng-model-options="{ trackBy: '$value.id' }" for objects
- [ ] mdOption with value
- [ ] Usage with input inside
- [ ] Usage with md-multiple

### TODO - POST RC1 ###
- [ ] Abstract placement logic in $mdSelect service to $mdMenu service

***************************************************/

var SELECT_EDGE_MARGIN = 8;
var selectNextId = 0;

angular.module('material.components.select', [
  'material.core',
  'material.components.backdrop'
])
.directive('mdSelect', SelectDirective)
.directive('mdSelectMenu', SelectMenuDirective)
.directive('mdOption', OptionDirective)
.directive('mdOptgroup', OptgroupDirective)
.provider('$mdSelect', SelectProvider);


/**
 * @ngdoc directive
 * @name mdSelect
 * @restrict E
 * @module material.components.select
 *
 * @description Displays a select box, bound to an ng-model.
 *
 * @param {expression} ng-model The model!
 * @param {boolean=} multiple Whether it's multiple.
 * @param {string=} placeholder Placeholder hint text.
 *
 * @usage
 * With a placeholder (label is added dynamically)
 * <hljs lang="html">
 *   <md-select
 *     ng-model="someModel"
 *     placeholder="Select a state">
 *     <md-option ng-value="opt" ng-repeat="opt in neighborhoods2">{{ opt }}</md-option>
 *   </md-select>
 * </hljs>
 *
 * With an explicit label
 * <hljs lang="html">
 *   <md-select
 *     ng-model="someModel">
 *     <md-select-label>Select a state</md-select-label>
 *     <md-option ng-value="opt" ng-repeat="opt in neighborhoods2">{{ opt }}</md-option>
 *   </md-select>
 * </hljs>
 */
function SelectDirective($mdSelect, $mdUtil, $mdTheming, $interpolate, $compile, $parse) {
  var intStart = $interpolate.startSymbol();
  var intEnd = $interpolate.endSymbol();

  return {
    restrict: 'E',
    require: '?ngModel',
    compile: compile
  };

  function compile(element, attr) {
    // The user is allowed to provide a label for the select as md-select-label child
    var labelEl = element.find('md-select-label').remove();

    // If not provided, we automatically make one
    if (!labelEl.length) {
      labelEl = angular.element('<md-select-label><span></span></md-select-label>');
    }
    labelEl.append('<span class="md-select-icon" aria-hidden="true"></span>');
    labelEl.addClass('md-select-label');
    labelEl.addClass(intStart + attr.ngModel + ' !== undefined ? \'\' : \'md-placeholder\'' + intEnd);
    labelEl.attr('id', 'select_label_' + $mdUtil.nextUid());

    // There's got to be an md-content inside. If there's not one, let's add it.
    if (!element.find('md-content').length) {
      element.append( angular.element('<md-content>').append(element.contents()) );
    }

    // Add progress spinner for md-options-loading
    if (attr.mdOnOpen) {
      element.find('md-content').prepend(
        angular.element('<md-progress-circular>')
               .attr('md-mode', 'indeterminate')
               .attr('ng-hide', '$$loadingAsyncDone')
               .wrap('<div>')
               .parent()
      );
    }

    // Use everything that's left inside element.contents() as the contents of the menu
    var selectTemplate = angular.element(
      '<div class="md-select-menu-container">' +
        '<md-select-menu ' +
        (angular.isDefined(attr.multiple) ? 'multiple' : '') + '>' +
          element.html() +
        '</md-select-menu></div>');

    element.empty().append(labelEl);

    $mdTheming(element);

    return function postLink(scope, element, attr, ngModel) {
      var isOpen;
      var labelEl = element.find('md-select-label');
      var customLabel = labelEl.text().length !== 0;
      if (!customLabel) labelEl = labelEl.children().eq(0);
      setInitialLabelValue();

      attr.$observe('disabled', function(disabled) {
        if (disabled !== undefined) {
          element.attr('tabindex', -1);
          element.off('click', openSelect);
          element.off('keydown', openOnKeypress);
        } else {
          element.attr('tabindex', 0);
          element.on('click', openSelect);
          element.on('keydown', openOnKeypress);
        }
      });

      if (attr.disabled === undefined) {
        element.on('click', openSelect);
        element.on('keydown', openOnKeypress);
      }

      element.attr({
        'role': 'combobox',
        'id': 'select_' + $mdUtil.nextUid(),
        'aria-haspopup': true,
        'aria-expanded': 'false',
        'aria-labelledby': labelEl.attr('id')
      });

      scope.$on('$destroy', function() {
        if (isOpen) {
          $mdSelect.cancel();
        }
      });

      // Create a fake select to find out the label value
      function setInitialLabelValue() {
        if ($parse(attr.ngModel)(scope)) {
          var fakeSelectEl = angular.element(selectTemplate).find('md-select-menu');
          fakeSelectEl.data('$ngModelController', ngModel);
          var fakeSelectScope = scope.$new();
          fakeSelectEl = $compile(fakeSelectEl)(fakeSelectScope);
          var fakeSelectCtrl = fakeSelectEl.controller('mdSelectMenu');
          fakeSelectScope.$$postDigest(function() {
            ngModel.$render();
            setLabelText(fakeSelectCtrl.selectedLabels());
            fakeSelectEl.scope().$destroy();
          });
        } else {
          setLabelText();
        }
      }

      function setLabelText(text) {
        if (customLabel) return; // Assume that user is handling it on their own
        var newText = text || attr.placeholder;
        labelEl.html(newText);
      }

      function openOnKeypress(e) {
        var allowedCodes = [32, 13, 38, 40];
        if (allowedCodes.indexOf(e.keyCode) != -1 ) {
          // prevent page scrolling on interaction
          e.preventDefault();
          openSelect(e);
        }
      }

      function openSelect() {
        scope.$evalAsync(function() {
          var selectEl = selectTemplate.clone();
          selectEl.find('md-select-menu').data('$ngModelController', ngModel);
          isOpen = true;
          $mdSelect.show({
            scope: scope.$new(),
            element: selectEl,
            target: element[0],
            hasBackdrop: true,
            loadingAsync: attr.mdOnOpen ? scope.$eval(attr.mdOnOpen) : false,
            setLabelText: setLabelText
          }).then(function(selectedText) {
            isOpen = false;
          });
        });
      }
    };

  }
}
SelectDirective.$inject = ["$mdSelect", "$mdUtil", "$mdTheming", "$interpolate", "$compile", "$parse"];

function SelectMenuDirective($parse, $mdUtil, $mdTheming) {

  SelectMenuController.$inject = ["$scope", "$attrs", "$element"];
  return {
    restrict: 'E',
    require: ['mdSelectMenu', '?ngModel'],
    controller: SelectMenuController,
    link: { pre: preLink }
  };

  // We use preLink instead of postLink to ensure that the select is initialized before
  // its child options run postLink.
  function preLink(scope, element, attr, ctrls) {
    var selectCtrl = ctrls[0];
    var ngModel = ctrls[1];

    $mdTheming(element);
    element.on('click', clickListener);
    element.on('keypress', keyListener);
    if (ngModel) selectCtrl.init(ngModel);
    configureAria();

    function configureAria() {
      element.attr({
        'id': 'select_menu_' + $mdUtil.nextUid(),
        'role': 'listbox',
        'aria-multiselectable': (selectCtrl.isMultiple ? 'true' : 'false')
      });
    }

    function keyListener(e) {
      if (e.keyCode == 13 || e.keyCode == 32) {
        clickListener(e);
      }
    }

    function clickListener(ev) {
      var option = $mdUtil.getClosest(ev.target, 'md-option');
      var optionCtrl = option && angular.element(option).data('$mdOptionController');
      if (!option || !optionCtrl) return;

      var optionHashKey = selectCtrl.hashGetter(optionCtrl.value);
      var isSelected = angular.isDefined(selectCtrl.selected[optionHashKey]);

      scope.$apply(function() {
        if (selectCtrl.isMultiple) {
          if (isSelected) {
            selectCtrl.deselect(optionHashKey);
          } else {
            selectCtrl.select(optionHashKey, optionCtrl.value);
          }
        } else {
          if (!isSelected) {
            selectCtrl.deselect( Object.keys(selectCtrl.selected)[0] );
            selectCtrl.select( optionHashKey, optionCtrl.value );
          }
        }
        selectCtrl.refreshViewValue();
      });
    }
  }



  function SelectMenuController($scope, $attrs, $element) {
    var self = this;
    self.isMultiple = angular.isDefined($attrs.multiple);
    // selected is an object with keys matching all of the selected options' hashed values
    self.selected = {};
    // options is an object with keys matching every option's hash value,
    // and values matching every option's controller.
    self.options = {};


    self.init = function(ngModel) {
      self.ngModel = ngModel;

      // Allow users to provide `ng-model="foo" ng-model-options="{trackBy: 'foo.id'}"` so
      // that we can properly compare objects set on the model to the available options
      if (ngModel.$options && ngModel.$options.trackBy) {
        var trackByLocals = {};
        var trackByParsed = $parse(ngModel.$options.trackBy);
        self.hashGetter = function(value, valueScope) {
          trackByLocals.$value = value;
          return trackByParsed(valueScope || $scope, trackByLocals);
        };
      // If the user doesn't provide a trackBy, we automatically generate an id for every
      // value passed in
      } else {
        self.hashGetter = function getHashValue(value) {
          if (angular.isObject(value)) {
            return '$$object_' + (value.$$mdSelectId || (value.$$mdSelectId = ++selectNextId));
          }
          return value;
        };
      }

      if (self.isMultiple) {
        ngModel.$validators['md-multiple'] = validateArray;
        ngModel.$render = renderMultiple;

        // watchCollection on the model because by default ngModel only watches the model's
        // reference. This allowed the developer to also push and pop from their array.
        $scope.$watchCollection($attrs.ngModel, function(value) {
          if (validateArray(value)) renderMultiple(value);
        });
      } else {
        ngModel.$render = renderSingular;
      }

      function validateArray(modelValue, viewValue) {
        // If a value is truthy but not an array, reject it.
        // If value is undefined/falsy, accept that it's an empty array.
        return angular.isArray(modelValue || viewValue || []);
      }
    };

    self.selectedLabels = function() {
      var selectedOptionEls = nodesToArray($element[0].querySelectorAll('md-option[selected]'));
      if (selectedOptionEls.length) {
        return selectedOptionEls.map(function(el) { return el.textContent; }).join(', ');
      } else {
        return '';
      }
    };

    self.select = function(hashKey, hashedValue) {
      var option = self.options[hashKey];
      option && option.setSelected(true);
      self.selected[hashKey] = hashedValue;
    };
    self.deselect = function(hashKey) {
      var option = self.options[hashKey];
      option && option.setSelected(false);
      delete self.selected[hashKey];
    };

    self.addOption = function(hashKey, optionCtrl) {
      if (angular.isDefined(self.options[hashKey])) {
        throw new Error('Duplicate md-option values are not allowed in a select. ' +
                        'Duplicate value "' + optionCtrl.value + '" found.');
      }
      self.options[hashKey] = optionCtrl;

      // If this option's value was already in our ngModel, go ahead and select it.
      if (angular.isDefined(self.selected[hashKey])) {
        self.select(hashKey, optionCtrl.value);
        self.refreshViewValue();
      }
    };
    self.removeOption = function(hashKey) {
      delete self.options[hashKey];
      // Don't deselect an option when it's removed - the user's ngModel should be allowed
      // to have values that do not match a currently available option.
    };

    self.refreshViewValue = function() {
      var values = [];
      var option;
      for (var hashKey in self.selected) {
         // If this hashKey has an associated option, push that option's value to the model.
         if ((option = self.options[hashKey])) {
           values.push(option.value);
         } else {
           // Otherwise, the given hashKey has no associated option, and we got it
           // from an ngModel value at an earlier time. Push the unhashed value of
           // this hashKey to the model.
           // This allows the developer to put a value in the model that doesn't yet have
           // an associated option.
           values.push(self.selected[hashKey]);
         }
      }
      self.ngModel.$setViewValue(self.isMultiple ? values : values[0]);
    };

    function renderMultiple() {
      var newSelectedValues = self.ngModel.$modelValue || self.ngModel.$viewValue;
      if (!angular.isArray(newSelectedValues)) return;

      var oldSelected = Object.keys(self.selected);

      var newSelectedHashes = newSelectedValues.map(self.hashGetter);
      var deselected = oldSelected.filter(function(hash) {
        return newSelectedHashes.indexOf(hash) === -1;
      });

      deselected.forEach(self.deselect);
      newSelectedHashes.forEach(function(hashKey, i) {
        self.select(hashKey, newSelectedValues[i]);
      });
    }
    function renderSingular() {
      var value = self.ngModel.$viewValue || self.ngModel.$modelValue;
      Object.keys(self.selected).forEach(self.deselect);
      self.select( self.hashGetter(value), value );
    }
  }

}
SelectMenuDirective.$inject = ["$parse", "$mdUtil", "$mdTheming"];

function OptionDirective($mdInkRipple, $mdUtil) {

  OptionController.$inject = ["$element"];
  return {
    restrict: 'E',
    require: ['mdOption', '^^mdSelectMenu'],
    controller: OptionController,
    compile: compile
  };

  function compile(element, attr) {
    // Manual transclusion to avoid the extra inner <span> that ng-transclude generates
    element.append( angular.element('<div class="md-text">').append(element.contents()) );
    if (attr.tabindex === undefined) element.attr('tabindex', 0);
    return postLink;
  }

  function postLink(scope, element, attr, ctrls) {
    var optionCtrl = ctrls[0];
    var selectCtrl = ctrls[1];

    if (angular.isDefined(attr.ngValue)) {
      scope.$watch(attr.ngValue, setOptionValue);
    } else if (angular.isDefined(attr.value)) {
      setOptionValue(attr.value);
    } else {
      throw new Error("Expected either ngValue or value attr");
    }

    $mdInkRipple.attachButtonBehavior(scope, element);
    configureAria();

    function setOptionValue(newValue, oldValue) {
      var oldHashKey = selectCtrl.hashGetter(oldValue, scope);
      var newHashKey = selectCtrl.hashGetter(newValue, scope);

      optionCtrl.hashKey = newHashKey;
      optionCtrl.value = newValue;

      selectCtrl.removeOption(oldHashKey, optionCtrl);
      selectCtrl.addOption(newHashKey, optionCtrl);
    }

    scope.$on('$destroy', function() {
      selectCtrl.removeOption(optionCtrl.hashKey, optionCtrl);
    });

    function configureAria() {
      element.attr({
        'role': 'option',
        'aria-selected': 'false',
        'id': 'select_option_'+ $mdUtil.nextUid()
      });
    }
  }

  function OptionController($element) {
    this.selected = false;
    this.setSelected = function(isSelected) {
      if (isSelected && !this.selected) {
        $element.attr({
          'selected': 'selected',
          'aria-selected': 'true'
        });
      } else if (!isSelected && this.selected) {
        $element.removeAttr('selected');
        $element.attr('aria-selected', 'false');
      }
      this.selected = isSelected;
    };
  }

}
OptionDirective.$inject = ["$mdInkRipple", "$mdUtil"];

function OptgroupDirective() {
  return {
    restrict: 'E',
    compile: compile
  };
  function compile(el, attrs) {
    var labelElement = el.find('label');
    if (!labelElement.length) {
      labelElement = angular.element('<label>');
      el.prepend(labelElement);
    }
    if (attrs.label) labelElement.text(attrs.label);
  }
}

function SelectProvider($$interimElementProvider) {
  selectDefaultOptions.$inject = ["$mdSelect", "$mdConstant", "$$rAF", "$mdUtil", "$mdTheming", "$timeout"];
  return $$interimElementProvider('$mdSelect')
    .setDefaults({
      methods: ['target'],
      options: selectDefaultOptions
    });

  /* @ngInject */
  function selectDefaultOptions($mdSelect, $mdConstant, $$rAF, $mdUtil, $mdTheming, $timeout) {
    return {
      parent: 'body',
      onShow: onShow,
      onRemove: onRemove,
      hasBackdrop: true,
      disableParentScroll: $mdUtil.floatingScrollbars(),
      themable: true
    };

    function onShow(scope, element, opts) {
      if (!opts.target) {
        throw new Error('$mdSelect.show() expected a target element in options.target but got ' +
                        '"' + opts.target + '"!');
      }

      angular.extend(opts, {
        target: angular.element(opts.target), //make sure it's not a naked dom node
        parent: angular.element(opts.parent),
        selectEl: element.find('md-select-menu'),
        contentEl: element.find('md-content'),
        backdrop: opts.hasBackdrop && angular.element('<md-backdrop class="md-select-backdrop">')
      });
    
      var optionNodes = [];

      configureAria();

      if (opts.loadingAsync && opts.loadingAsync.then) {
        opts.loadingAsync.then(function() {
          scope.$$loadingAsyncDone = true;
          // Give ourselves two frames for the progress loader to clear out.
          $$rAF(function() {
            $$rAF(function() {
              // Don't go forward if the select has been removed in this time...
              if (opts.isRemoved) return;
              animateSelect(scope, element, opts);
              optionNodes = nodesToArray(opts.selectEl[0].getElementsByTagName('md-option'));
            });
          });
        });
      }

      if (opts.disableParentScroll) {
        opts.disableTarget = opts.parent.find('md-content');
        if (!opts.disableTarget.length) opts.disableTarget = opts.parent;
        opts.lastOverflow = opts.disableTarget.css('overflow');
        opts.disableTarget.css('overflow', 'hidden');
      }

      // Only activate click listeners after a short time to stop accidental double taps/clicks
      // from clicking the wrong item
      $timeout(activateInteraction, 75, false);

      if (opts.backdrop) {
        $mdTheming.inherit(opts.backdrop, opts.parent);
        opts.parent.append(opts.backdrop);
      }
      opts.parent.append(element);

      // Give the select a frame to 'initialize' in the DOM,
      // so we can read its height/width/position
      $$rAF(function() {
        $$rAF(function() {
          if (opts.isRemoved) return;
          animateSelect(scope, element, opts);
          optionNodes = nodesToArray(element[0].querySelectorAll('md-option'));
        });
      });

      return $mdUtil.transitionEndPromise(opts.selectEl);

      function configureAria() {
        opts.selectEl.attr('aria-labelledby', opts.target.attr('id'));
        opts.target.attr('aria-owns', opts.selectEl.attr('id'));
        opts.target.attr('aria-expanded', 'true');
      }

      function activateInteraction() {
        if (opts.isRemoved) return;
        var selectCtrl = opts.selectEl.controller('mdSelectMenu') || {};
        element.addClass('md-clickable');

        opts.backdrop && opts.backdrop.on('click', function(e) {
          e.preventDefault();
          e.stopPropagation();
          opts.restoreFocus = false;
          scope.$apply($mdSelect.cancel);
        });

        // Escape to close
        opts.selectEl.on('keydown', function(e) {
          switch (e.keyCode) {
            case $mdConstant.KEY_CODE.TAB:
            case $mdConstant.KEY_CODE.ESCAPE:
              e.preventDefault();
              opts.restoreFocus = true;
              scope.$apply($mdSelect.cancel);
          }
        });

        // Cycling of options, and closing on enter
        opts.selectEl.on('keydown', function(e) {
          switch (e.keyCode) {
            case $mdConstant.KEY_CODE.UP_ARROW: return focusPrevOption();
            case $mdConstant.KEY_CODE.DOWN_ARROW: return focusNextOption();
          }
        });

        function focusNextOption() {
          var index;
          if ((index = optionNodes.indexOf(opts.focusedNode)) == -1) {
            // We lost the previously focused element, reset to middle
            index = Math.floor( (optionNodes.length - 1) / 2 );
          } else {
            if (index < optionNodes.length - 1) ++index;
          }
          opts.focusedNode = optionNodes[index];
          optionNodes[index].focus();
        }
        function focusPrevOption() {
          var index;
          if ((index = optionNodes.indexOf(opts.focusedNode)) == -1) {
            // We lost the previously focused element, reset to middle
            index = Math.floor( (optionNodes.length - 1) / 2 );
          } else {
            if (index > 0) --index;
          }
          opts.focusedNode = optionNodes[index];
          optionNodes[index].focus();
        }

        if (!selectCtrl.isMultiple) {
          opts.selectEl.on('click', closeMenu);
          opts.selectEl.on('keydown', function(e) {
            if (e.keyCode == 32 || e.keyCode == 13) {
              closeMenu();
            }
          });
        }
        function closeMenu() {
          opts.restoreFocus = true;
          scope.$evalAsync(function() {
            $mdSelect.hide(selectCtrl.ngModel.$viewValue);
          });
        }
      }

    }

    function onRemove(scope, element, opts) {
      opts.isRemoved = true;
      element.addClass('md-leave').removeClass('md-clickable');
      opts.target.attr('aria-expanded', 'false');

      if (opts.disableParentScroll && $mdUtil.floatingScrollbars()) {
        opts.disableTarget.css('overflow', opts.lastOverflow);
        delete opts.lastOverflow;
        delete opts.disableTarget;
      }

      opts.setLabelText && opts.setLabelText(opts.selectEl.controller('mdSelectMenu').selectedLabels());

      return $mdUtil.transitionEndPromise(element).then(function() {
        element.remove();
        opts.backdrop && opts.backdrop.remove();
        if (opts.restoreFocus) opts.target.focus();
      });
    }

    function animateSelect(scope, element, opts) {
      var containerNode = element[0],
          targetNode = opts.target[0],
          parentNode = opts.parent[0],
          selectNode = opts.selectEl[0],
          contentNode = opts.contentEl[0],
          parentRect = parentNode.getBoundingClientRect(),
          targetRect = $mdUtil.clientRect(targetNode, parentNode),
          shouldOpenAroundTarget = false,
          bounds = {
            left: parentNode.scrollLeft + SELECT_EDGE_MARGIN,
            top: parentNode.scrollTop + SELECT_EDGE_MARGIN,
            bottom: parentRect.height + parentNode.scrollTop - SELECT_EDGE_MARGIN,
            right: parentRect.width - parentNode.scrollLeft - SELECT_EDGE_MARGIN
          },
          spaceAvailable = {
            top: targetRect.top - bounds.top,
            left: targetRect.left - bounds.left,
            right: bounds.right - (targetRect.left + targetRect.width),
            bottom: bounds.bottom - (targetRect.top + targetRect.height)
          },
          maxWidth = parentRect.width - SELECT_EDGE_MARGIN * 2,
          isScrollable = contentNode.scrollHeight > contentNode.offsetHeight,
          selectedNode = selectNode.querySelector('md-option[selected]'),
          optionNodes = nodesToArray(selectNode.getElementsByTagName('md-option')),
          optgroupNodes = selectNode.getElementsByTagName('md-optgroup');

      var centeredNode;
      // If a selected node, center around that
      if (selectedNode) {
        centeredNode = selectedNode;
      // If there are option groups, center around the first option
      } else if (optgroupNodes.length) {
        centeredNode = optionNodes[0];
      // Otherwise, lets center on the middle optionNode
      } else if (optionNodes.length){
        centeredNode = optionNodes[Math.floor(optionNodes.length / 2 )];
      // In case there are no options, center on whatevers in there... (such as a progress indicator)
      } else {
        centeredNode = contentNode.firstElementChild || contentNode;
      }

      if (contentNode.offsetWidth > maxWidth) {
        contentNode.style['max-width'] = maxWidth + 'px';
      }
      if (shouldOpenAroundTarget) {
        contentNode.style['min-width'] = targetRect.width + 'px';
      }

      // Remove padding before we compute the position of the menu
      if (isScrollable) {
        selectNode.classList.add('md-overflow');
      }

      // Get the selectMenuRect *after* max-width is possibly set above
      var selectMenuRect = selectNode.getBoundingClientRect();
      var centeredRect = getOffsetRect(centeredNode);

      if (centeredNode) {
        var centeredStyle = window.getComputedStyle(centeredNode);
        centeredRect.paddingLeft = parseInt(centeredStyle['padding-left'], 10);
        centeredRect.paddingRight = parseInt(centeredStyle['padding-right'], 10);
      }

      var focusedNode = centeredNode || optionNodes[0];
      if (focusedNode) {
        opts.focusedNode = focusedNode;
        focusedNode.focus();
      }

      if (isScrollable) {
        var scrollBuffer = contentNode.offsetHeight / 2;
        contentNode.scrollTop = centeredRect.top + centeredRect.height / 2 - scrollBuffer;

        if (spaceAvailable.top < scrollBuffer) {
          contentNode.scrollTop = Math.min(
            centeredRect.top,
            contentNode.scrollTop + scrollBuffer - spaceAvailable.top
          );
        } else if (spaceAvailable.bottom < scrollBuffer) {
          contentNode.scrollTop = Math.max(
            centeredRect.top + centeredRect.height - selectMenuRect.height,
            contentNode.scrollTop - scrollBuffer + spaceAvailable.bottom
          );
        }
      }

      var left, top, transformOrigin;
      if (shouldOpenAroundTarget) {
        left = targetRect.left;
        top = targetRect.top + targetRect.height;
        transformOrigin = '50% 0';
        if (top + selectMenuRect.height > bounds.bottom) {
          top = targetRect.top - selectMenuRect.height;
          transformOrigin = '50% 100%';
        }
      } else {
        left = targetRect.left + centeredRect.left - centeredRect.paddingLeft;
        top = targetRect.top + targetRect.height / 2 - centeredRect.height / 2 -
          centeredRect.top + contentNode.scrollTop;
        transformOrigin = (centeredRect.left + targetRect.width / 2) + 'px ' +
        (centeredRect.top + centeredRect.height / 2 - contentNode.scrollTop) + 'px 0px';
        containerNode.style['min-width'] = targetRect.width + centeredRect.paddingLeft + centeredRect.paddingRight + 'px';
      }

      // Keep left and top within the window
      containerNode.style.left = clamp(bounds.left, left, bounds.right) + 'px';
      containerNode.style.top = clamp(bounds.top, top, bounds.bottom) + 'px';
      selectNode.style[$mdConstant.CSS.TRANSFORM_ORIGIN] = transformOrigin;

      selectNode.style[$mdConstant.CSS.TRANSFORM] = 'scale(' +
        Math.min(targetRect.width / selectMenuRect.width, 1.0) + ',' +
        Math.min(targetRect.height / selectMenuRect.height, 1.0) +
      ')';

      $$rAF(function() {
        element.addClass('md-active');
        selectNode.style[$mdConstant.CSS.TRANSFORM] = '';
      });
    }

  }

  function clamp(min, n, max) {
    return Math.min(max, Math.max(n, min));
  }

  function getOffsetRect(node) {
    return node ? {
      left: node.offsetLeft,
      top: node.offsetTop,
      width: node.offsetWidth,
      height: node.offsetHeight
    } : { left: 0, top: 0, width: 0, height: 0 };
  }
}
SelectProvider.$inject = ["$$interimElementProvider"];

// Annoying method to copy nodes to an array, thanks to IE
function nodesToArray(nodes) {
  var results = [];
  for (var i = 0; i < nodes.length; ++i) {
    results.push(nodes.item(i));
  }
  return results;
}
})();


/*!
 * Angular Material Design
 * https://github.com/angular/material
 * @license MIT
 * v0.7.0-rc3
 */
(function() {
'use strict';

/**
 * @ngdoc module
 * @name material.components.tabs
 * @description
 *
 *  Tabs, created with the `<md-tabs>` directive provide *tabbed* navigation with different styles.
 *  The Tabs component consists of clickable tabs that are aligned horizontally side-by-side.
 *
 *  Features include support for:
 *
 *  - static or dynamic tabs,
 *  - responsive designs,
 *  - accessibility support (ARIA),
 *  - tab pagination,
 *  - external or internal tab content,
 *  - focus indicators and arrow-key navigations,
 *  - programmatic lookup and access to tab controllers, and
 *  - dynamic transitions through different tab contents.
 *
 */
/*
 * @see js folder for tabs implementation
 */
angular.module('material.components.tabs', [
  'material.core'
]);
})();

(function() {
'use strict';

/**
 * Conditionally configure ink bar animations when the
 * tab selection changes. If `mdNoBar` then do not show the
 * bar nor animate.
 */
angular.module('material.components.tabs')
  .directive('mdTabsInkBar', MdTabInkDirective);

function MdTabInkDirective($$rAF) {

  var lastIndex = 0;

  return {
    restrict: 'E',
    require: ['^?mdNoBar', '^mdTabs'],
    link: postLink
  };

  function postLink(scope, element, attr, ctrls) {
    if (ctrls[0]) return;

    var tabsCtrl = ctrls[1],
        debouncedUpdateBar = $$rAF.debounce(updateBar);

    tabsCtrl.inkBarElement = element;

    scope.$on('$mdTabsPaginationChanged', debouncedUpdateBar);

    function updateBar() {
      var selected = tabsCtrl.getSelectedItem();
      var hideInkBar = !selected || tabsCtrl.count() < 2;

      element.css('display', hideInkBar ? 'none' : 'block');

      if (hideInkBar) return;

      if (scope.pagination && scope.pagination.tabData) {
        var index = tabsCtrl.getSelectedIndex();
        var data = scope.pagination.tabData.tabs[index] || { left: 0, right: 0, width: 0 };
        var right = element.parent().prop('offsetWidth') - data.right;
        var classNames = ['md-transition-left', 'md-transition-right', 'md-no-transition'];
        var classIndex = lastIndex > index ? 0 : lastIndex < index ? 1 : 2;

        element
            .removeClass(classNames.join(' '))
            .addClass(classNames[classIndex])
            .css({ left: (data.left + 1) + 'px', right: right + 'px' });

        lastIndex = index;
      }
    }
  }
}
MdTabInkDirective.$inject = ["$$rAF"];
})();

(function() {
'use strict';

angular.module('material.components.tabs')
    .directive('mdTabsPagination', TabPaginationDirective);

function TabPaginationDirective($mdConstant, $window, $$rAF, $$q, $timeout, $mdMedia) {

  // Must match (2 * width of paginators) in scss
  var PAGINATORS_WIDTH = (8 * 4) * 2;

  return {
    restrict: 'A',
    require: '^mdTabs',
    link: postLink
  };

  function postLink(scope, element, attr, tabsCtrl) {

    var tabs = element[0].getElementsByTagName('md-tab');
    var debouncedUpdatePagination = $$rAF.debounce(updatePagination);
    var tabsParent = element.children();
    var state = scope.pagination = {
      page: -1,
      active: false,
      clickNext: function() { userChangePage(+1); },
      clickPrevious: function() { userChangePage(-1); }
    };

    scope.$on('$mdTabsChanged', debouncedUpdatePagination);
    angular.element($window).on('resize', debouncedUpdatePagination);

    scope.$on('$destroy', function() {
      angular.element($window).off('resize', debouncedUpdatePagination);
    });

    scope.$watch(function() { return tabsCtrl.tabToFocus; }, onTabFocus);

    // Make sure we don't focus an element on the next page
    // before it's in view
    function onTabFocus(tab, oldTab) {
      if (!tab) return;

      var pageIndex = getPageForTab(tab);
      if (!state.active || pageIndex === state.page) {
        tab.element.focus();
      } else {
        // Go to the new page, wait for the page transition to end, then focus.
        oldTab && oldTab.element.blur();
        setPage(pageIndex).then(function() { tab.element.focus(); });
      }
    }

    // Called when page is changed by a user action (click)
    function userChangePage(increment) {
      var sizeData = state.tabData;
      var newPage = Math.max(0, Math.min(sizeData.pages.length - 1, state.page + increment));
      var newTabIndex = sizeData.pages[newPage][ increment > 0 ? 'firstTabIndex' : 'lastTabIndex' ];
      var newTab = tabsCtrl.itemAt(newTabIndex);
      onTabFocus(newTab);
    }

    function updatePagination() {
      if (!element.prop('offsetParent')) {
        var watcher = waitForVisible();
        return;
      }

      var tabs = element.find('md-tab');

      disablePagination();

      var sizeData = state.tabData = calculateTabData();
      var needPagination = state.active = sizeData.pages.length > 1;

      if (needPagination) { enablePagination(); }

      scope.$evalAsync(function () { scope.$broadcast('$mdTabsPaginationChanged'); });

      function enablePagination() {
        tabsParent.css('width', '9999px');

        //-- apply filler margins
        angular.forEach(sizeData.tabs, function (tab) {
          angular.element(tab.element).css('margin-left', tab.filler + 'px');
        });

        setPage(getPageForTab(tabsCtrl.getSelectedItem()));
      }

      function disablePagination() {
        slideTabButtons(0);
        tabsParent.css('width', '');
        tabs.css('width', '');
        tabs.css('margin-left', '');
        state.page = null;
        state.active = false;
      }

      function waitForVisible() {
        return watcher || scope.$watch(
            function () {
              $timeout(function () {
                if (element[0].offsetParent) {
                  if (angular.isFunction(watcher)) {
                    watcher();
                  }
                  debouncedUpdatePagination();
                  watcher = null;
                }
              }, 0, false);
            }
        );
      }
    }

    function slideTabButtons(x) {
      if (tabsCtrl.pagingOffset === x) {
        // Resolve instantly if no change
        return $$q.when();
      }

      var deferred = $$q.defer();

      tabsCtrl.$$pagingOffset = x;
      tabsParent.css($mdConstant.CSS.TRANSFORM, 'translate3d(' + x + 'px,0,0)');
      tabsParent.on($mdConstant.CSS.TRANSITIONEND, onTabsParentTransitionEnd);

      return deferred.promise;

      function onTabsParentTransitionEnd(ev) {
        // Make sure this event didn't bubble up from an animation in a child element.
        if (ev.target === tabsParent[0]) {
          tabsParent.off($mdConstant.CSS.TRANSITIONEND, onTabsParentTransitionEnd);
          deferred.resolve();
        }
      }
    }

    function shouldStretchTabs() {
      switch (scope.stretchTabs) {
        case 'never':  return false;
        case 'always': return true;
        default:       return $mdMedia('sm');
      }
    }

    function calculateTabData(noAdjust) {
      var clientWidth = element.parent().prop('offsetWidth');
      var tabsWidth = clientWidth - PAGINATORS_WIDTH - 1;
      var $tabs = angular.element(tabs);
      var totalWidth = 0;
      var max = 0;
      var tabData = [];
      var pages = [];
      var currentPage;

      $tabs.css('max-width', '');
      angular.forEach(tabs, function (tab, index) {
        var tabWidth = Math.min(tabsWidth, tab.offsetWidth);
        var data = {
          element: tab,
          left: totalWidth,
          width: tabWidth,
          right: totalWidth + tabWidth,
          filler: 0
        };

        //-- This calculates the page for each tab.  The first page will use the clientWidth, which
        //   does not factor in the pagination items.  After the first page, tabsWidth is used
        //   because at this point, we know that the pagination buttons will be shown.
        data.page = Math.ceil(data.right / ( pages.length === 1 && index === tabs.length - 1 ? clientWidth : tabsWidth )) - 1;

        if (data.page >= pages.length) {
          data.filler = (tabsWidth * data.page) - data.left;
          data.right += data.filler;
          data.left += data.filler;
          currentPage = {
            left: data.left,
            firstTabIndex: index,
            lastTabIndex: index,
            tabs: [ data ]
          };
          pages.push(currentPage);
        } else {
          currentPage.lastTabIndex = index;
          currentPage.tabs.push(data);
        }
        totalWidth = data.right;
        max = Math.max(max, tabWidth);
        tabData.push(data);
      });
      $tabs.css('max-width', tabsWidth + 'px');

      if (!noAdjust && shouldStretchTabs()) {
        return adjustForStretchedTabs();
      } else {
        return {
          width: totalWidth,
          max: max,
          tabs: tabData,
          pages: pages,
          tabElements: tabs
        };
      }


      function adjustForStretchedTabs() {
        var canvasWidth = pages.length === 1 ? clientWidth : tabsWidth;
        var tabsPerPage = Math.min(Math.floor(canvasWidth / max), tabs.length);
        var tabWidth    = Math.floor(canvasWidth / tabsPerPage);
        $tabs.css('width', tabWidth + 'px');
        return calculateTabData(true);
      }
    }

    function getPageForTab(tab) {
      var tabIndex = tabsCtrl.indexOf(tab);
      if (tabIndex === -1) return 0;

      var sizeData = state.tabData;

      return sizeData ? sizeData.tabs[tabIndex].page : 0;
    }

    function setPage(page) {
      if (page === state.page) return;

      var lastPage = state.tabData.pages.length - 1;

      if (page < 0) page = 0;
      if (page > lastPage) page = lastPage;

      state.hasPrev = page > 0;
      state.hasNext = page < lastPage;

      state.page = page;

      scope.$broadcast('$mdTabsPaginationChanged');

      return slideTabButtons(-state.tabData.pages[page].left);
    }
  }

}
TabPaginationDirective.$inject = ["$mdConstant", "$window", "$$rAF", "$$q", "$timeout", "$mdMedia"];
})();

(function() {
'use strict';


angular.module('material.components.tabs')
  .controller('$mdTab', TabItemController);

function TabItemController($scope, $element, $attrs, $compile, $animate, $mdUtil, $parse, $timeout) {
  var self = this;

  // Properties
  self.contentContainer = angular.element('<div class="md-tab-content ng-hide">');
  self.hammertime = new Hammer(self.contentContainer[0]);
  self.element = $element;

  // Methods
  self.isDisabled = isDisabled;
  self.onAdd = onAdd;
  self.onRemove = onRemove;
  self.onSelect = onSelect;
  self.onDeselect = onDeselect;

  var disabledParsed = $parse($attrs.ngDisabled);
  function isDisabled() {
    return disabledParsed($scope.$parent);
  }
  
  /**
   * Add the tab's content to the DOM container area in the tabs,
   * @param contentArea the contentArea to add the content of the tab to
   */
  function onAdd(contentArea, shouldDisconnectScope) {
    if (self.content.length) {
      self.contentContainer.append(self.content);
      self.contentScope = $scope.$parent.$new();
      contentArea.append(self.contentContainer);

      $compile(self.contentContainer)(self.contentScope);
      if (shouldDisconnectScope === true) {
        $timeout(function () {
          $mdUtil.disconnectScope(self.contentScope);
        }, 0, false);
      }
    }
  }

  function onRemove() {
    self.hammertime.destroy();
    $animate.leave(self.contentContainer).then(function() {
      self.contentScope && self.contentScope.$destroy();
      self.contentScope = null;
    });
  }

  function toggleAnimationClass(rightToLeft) {
    self.contentContainer[rightToLeft ? 'addClass' : 'removeClass']('md-transition-rtl');
  }

  function onSelect(rightToLeft) {
    // Resume watchers and events firing when tab is selected
    $mdUtil.reconnectScope(self.contentScope);
    self.hammertime.on('swipeleft swiperight', $scope.onSwipe);

    $element.addClass('active');
    $element.attr('aria-selected', true);
    $element.attr('tabIndex', 0);
    toggleAnimationClass(rightToLeft);
    $animate.removeClass(self.contentContainer, 'ng-hide');

    $scope.onSelect();
  }

  function onDeselect(rightToLeft) {
    // Stop watchers & events from firing while tab is deselected
    $mdUtil.disconnectScope(self.contentScope);
    self.hammertime.off('swipeleft swiperight', $scope.onSwipe);

    $element.removeClass('active');
    $element.attr('aria-selected', false);
    // Only allow tabbing to the active tab
    $element.attr('tabIndex', -1);
    toggleAnimationClass(rightToLeft);
    $animate.addClass(self.contentContainer, 'ng-hide');

    $scope.onDeselect();
  }

}
TabItemController.$inject = ["$scope", "$element", "$attrs", "$compile", "$animate", "$mdUtil", "$parse", "$timeout"];

})();

(function() {
'use strict';

angular.module('material.components.tabs')
  .directive('mdTab', MdTabDirective);

/**
 * @ngdoc directive
 * @name mdTab
 * @module material.components.tabs
 *
 * @restrict E
 *
 * @description
 * `<md-tab>` is the nested directive used [within `<md-tabs>`] to specify each tab with a **label** and optional *view content*.
 *
 * If the `label` attribute is not specified, then an optional `<md-tab-label>` tag can be used to specify more
 * complex tab header markup. If neither the **label** nor the **md-tab-label** are specified, then the nested
 * markup of the `<md-tab>` is used as the tab header markup.
 *
 * If a tab **label** has been identified, then any **non-**`<md-tab-label>` markup
 * will be considered tab content and will be transcluded to the internal `<div class="md-tabs-content">` container.
 *
 * This container is used by the TabsController to show/hide the active tab's content view. This synchronization is
 * automatically managed by the internal TabsController whenever the tab selection changes. Selection changes can
 * be initiated via data binding changes, programmatic invocation, or user gestures.
 *
 * @param {string=} label Optional attribute to specify a simple string as the tab label
 * @param {boolean=} md-active When evaluteing to true, selects the tab.
 * @param {boolean=} disabled If present, disabled tab selection.
 * @param {expression=} md-on-deselect Expression to be evaluated after the tab has been de-selected.
 * @param {expression=} md-on-select Expression to be evaluated after the tab has been selected.
 *
 *
 * @usage
 *
 * <hljs lang="html">
 * <md-tab label="" disabled="" md-on-select="" md-on-deselect="" >
 *   <h3>My Tab content</h3>
 * </md-tab>
 *
 * <md-tab >
 *   <md-tab-label>
 *     <h3>My Tab content</h3>
 *   </md-tab-label>
 *   <p>
 *     Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium,
 *     totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae
 *     dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit,
 *     sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt.
 *   </p>
 * </md-tab>
 * </hljs>
 *
 */
function MdTabDirective($mdInkRipple, $compile, $mdUtil, $mdConstant, $timeout) {
  return {
    restrict: 'E',
    require: ['mdTab', '^mdTabs'],
    controller: '$mdTab',
    scope: {
      onSelect: '&mdOnSelect',
      onDeselect: '&mdOnDeselect',
      label: '@'
    },
    compile: compile
  };

  function compile(element, attr) {
    var tabLabel = element.find('md-tab-label');

    if (tabLabel.length) {
      // If a tab label element is found, remove it for later re-use.
      tabLabel.remove();

    } else if (angular.isDefined(attr.label)) {
      // Otherwise, try to use attr.label as the label
      tabLabel = angular.element('<md-tab-label>').html(attr.label);

    } else {
      // If nothing is found, use the tab's content as the label
      tabLabel = angular.element('<md-tab-label>')
                        .append(element.contents().remove());
    }

    // Everything that's left as a child is the tab's content.
    var tabContent = element.contents().remove();

    return function postLink(scope, element, attr, ctrls) {

      var tabItemCtrl = ctrls[0]; // Controller for THIS tabItemCtrl
      var tabsCtrl = ctrls[1]; // Controller for ALL tabs

      scope.$watch(
          function () { return attr.label; },
          function () { $timeout(function () { tabsCtrl.scope.$broadcast('$mdTabsChanged'); }, 0, false); }
      );

      transcludeTabContent();
      configureAria();

      var detachRippleFn = $mdInkRipple.attachTabBehavior(scope, element, {
        colorElement: tabsCtrl.inkBarElement
      });
      tabsCtrl.add(tabItemCtrl);
      scope.$on('$destroy', function() {
        detachRippleFn();
        tabsCtrl.remove(tabItemCtrl);
      });
      element.on('$destroy', function () {
        //-- wait for item to be removed from the dom
        $timeout(function () {
          tabsCtrl.scope.$broadcast('$mdTabsChanged');
        }, 0, false);
      });

      if (!angular.isDefined(attr.ngClick)) {
        element.on('click', defaultClickListener);
      }
      element.on('keydown', keydownListener);
      scope.onSwipe = onSwipe;

      if (angular.isNumber(scope.$parent.$index)) {
        watchNgRepeatIndex();
      }
      if (angular.isDefined(attr.mdActive)) {
        watchActiveAttribute();
      }
      watchDisabled();

      function transcludeTabContent() {
        // Clone the label we found earlier, and $compile and append it
        var label = tabLabel.clone();
        element.append(label);
        $compile(label)(scope.$parent);

        // Clone the content we found earlier, and mark it for later placement into
        // the proper content area.
        tabItemCtrl.content = tabContent.clone();
      }

      //defaultClickListener isn't applied if the user provides an ngClick expression.
      function defaultClickListener() {
        scope.$apply(function() {
          tabsCtrl.select(tabItemCtrl);
          tabsCtrl.focus(tabItemCtrl);
        });
      }
      function keydownListener(ev) {
        if (ev.keyCode == $mdConstant.KEY_CODE.SPACE || ev.keyCode == $mdConstant.KEY_CODE.ENTER ) {
          // Fire the click handler to do normal selection if space is pressed
          element.triggerHandler('click');
          ev.preventDefault();
        } else if (ev.keyCode === $mdConstant.KEY_CODE.LEFT_ARROW) {
          scope.$evalAsync(function() {
            tabsCtrl.focus(tabsCtrl.previous(tabItemCtrl));
          });
        } else if (ev.keyCode === $mdConstant.KEY_CODE.RIGHT_ARROW) {
          scope.$evalAsync(function() {
            tabsCtrl.focus(tabsCtrl.next(tabItemCtrl));
          });
        }
      }

      function onSwipe(ev) {
        scope.$apply(function() {
          if (ev.type === 'swipeleft') {
            tabsCtrl.select(tabsCtrl.next());
          } else {
            tabsCtrl.select(tabsCtrl.previous());
          }
        });
      }

      // If tabItemCtrl is part of an ngRepeat, move the tabItemCtrl in our internal array
      // when its $index changes
      function watchNgRepeatIndex() {
        // The tabItemCtrl has an isolate scope, so we watch the $index on the parent.
        scope.$watch('$parent.$index', function $indexWatchAction(newIndex) {
          tabsCtrl.move(tabItemCtrl, newIndex);
        });
      }

      function watchActiveAttribute() {
        var unwatch = scope.$parent.$watch('!!(' + attr.mdActive + ')', activeWatchAction);
        scope.$on('$destroy', unwatch);

        function activeWatchAction(isActive) {
          var isSelected = tabsCtrl.getSelectedItem() === tabItemCtrl;

          if (isActive && !isSelected) {
            tabsCtrl.select(tabItemCtrl);
          } else if (!isActive && isSelected) {
            tabsCtrl.deselect(tabItemCtrl);
          }
        }
      }

      function watchDisabled() {
        scope.$watch(tabItemCtrl.isDisabled, disabledWatchAction);

        function disabledWatchAction(isDisabled) {
          element.attr('aria-disabled', isDisabled);

          // Auto select `next` tab when disabled
          var isSelected = (tabsCtrl.getSelectedItem() === tabItemCtrl);
          if (isSelected && isDisabled) {
            tabsCtrl.select(tabsCtrl.next() || tabsCtrl.previous());
          }

        }
      }

      function configureAria() {
        // Link together the content area and tabItemCtrl with an id
        var tabId = attr.id || ('tab_' + $mdUtil.nextUid());

        element.attr({
          id: tabId,
          role: 'tab',
          tabIndex: -1 //this is also set on select/deselect in tabItemCtrl
        });

        // Only setup the contentContainer's aria attributes if tab content is provided
        if (tabContent.length) {
          var tabContentId = 'content_' + tabId;
          if (!element.attr('aria-controls')) {
            element.attr('aria-controls', tabContentId);
          }
          tabItemCtrl.contentContainer.attr({
            id: tabContentId,
            role: 'tabpanel',
            'aria-labelledby': tabId
          });
        }
      }

    };

  }

}
MdTabDirective.$inject = ["$mdInkRipple", "$compile", "$mdUtil", "$mdConstant", "$timeout"];

})();

(function() {
'use strict';

angular.module('material.components.tabs')
  .controller('$mdTabs', MdTabsController);

function MdTabsController($scope, $element, $mdUtil, $timeout) {

  var tabsList = $mdUtil.iterator([], false);
  var self = this;

  // Properties
  self.$element = $element;
  self.scope = $scope;
  // The section containing the tab content $elements
  var contentArea = self.contentArea = angular.element($element[0].querySelector('.md-tabs-content'));

  // Methods from iterator
  var inRange = self.inRange = tabsList.inRange;
  var indexOf = self.indexOf = tabsList.indexOf;
  var itemAt = self.itemAt = tabsList.itemAt;
  self.count = tabsList.count;

  self.getSelectedItem = getSelectedItem;
  self.getSelectedIndex = getSelectedIndex;
  self.add = add;
  self.remove = remove;
  self.move = move;
  self.select = select;
  self.focus = focus;
  self.deselect = deselect;

  self.next = next;
  self.previous = previous;

  $scope.$on('$destroy', function() {
    deselect(getSelectedItem());
    for (var i = tabsList.count() - 1; i >= 0; i--) {
      remove(tabsList[i], true);
    }
  });

  // Get the selected tab
  function getSelectedItem() {
    return itemAt($scope.selectedIndex);
  }

  function getSelectedIndex() {
    return $scope.selectedIndex;
  }

  // Add a new tab.
  // Returns a method to remove the tab from the list.
  function add(tab, index) {
    tabsList.add(tab, index);

    // Select the new tab if we don't have a selectedIndex, or if the
    // selectedIndex we've been waiting for is this tab
    if (!angular.isDefined(tab.element.attr('md-active')) && ($scope.selectedIndex === -1 || !angular.isNumber($scope.selectedIndex) ||
        $scope.selectedIndex === self.indexOf(tab))) {
      tab.onAdd(self.contentArea, false);
      self.select(tab);
    } else {
      tab.onAdd(self.contentArea, true);
    }

    $scope.$broadcast('$mdTabsChanged');
  }

  function remove(tab, noReselect) {
    if (!tabsList.contains(tab)) return;
    if (noReselect) return;
    var isSelectedItem = getSelectedItem() === tab,
        newTab = previous() || next();

    deselect(tab);
    tabsList.remove(tab);
    tab.onRemove();

    $scope.$broadcast('$mdTabsChanged');

    if (isSelectedItem) { select(newTab); }
  }

  // Move a tab (used when ng-repeat order changes)
  function move(tab, toIndex) {
    var isSelected = getSelectedItem() === tab;

    tabsList.remove(tab);
    tabsList.add(tab, toIndex);
    if (isSelected) select(tab);

    $scope.$broadcast('$mdTabsChanged');
  }

  function select(tab, rightToLeft) {
    if (!tab || tab.isSelected || tab.isDisabled()) return;
    if (!tabsList.contains(tab)) return;

    if (!angular.isDefined(rightToLeft)) {
      rightToLeft = indexOf(tab) < $scope.selectedIndex;
    }
    deselect(getSelectedItem(), rightToLeft);

    $scope.selectedIndex = indexOf(tab);
    tab.isSelected = true;
    tab.onSelect(rightToLeft);

    $scope.$broadcast('$mdTabsChanged');
  }

  function focus(tab) {
    // this variable is watched by pagination
    self.tabToFocus = tab;
  }

  function deselect(tab, rightToLeft) {
    if (!tab || !tab.isSelected) return;
    if (!tabsList.contains(tab)) return;

    $scope.selectedIndex = -1;
    tab.isSelected = false;
    tab.onDeselect(rightToLeft);
  }

  function next(tab, filterFn) {
    return tabsList.next(tab || getSelectedItem(), filterFn || isTabEnabled);
  }
  function previous(tab, filterFn) {
    return tabsList.previous(tab || getSelectedItem(), filterFn || isTabEnabled);
  }

  function isTabEnabled(tab) {
    return tab && !tab.isDisabled();
  }

}
MdTabsController.$inject = ["$scope", "$element", "$mdUtil", "$timeout"];
})();

(function() {
'use strict';

angular.module('material.components.tabs')
  .directive('mdTabs', TabsDirective);

/**
 * @ngdoc directive
 * @name mdTabs
 * @module material.components.tabs
 *
 * @restrict E
 *
 * @description
 * The `<md-tabs>` directive serves as the container for 1..n `<md-tab>` child directives to produces a Tabs components.
 * In turn, the nested `<md-tab>` directive is used to specify a tab label for the **header button** and a [optional] tab view
 * content that will be associated with each tab button.
 *
 * Below is the markup for its simplest usage:
 *
 *  <hljs lang="html">
 *  <md-tabs>
 *    <md-tab label="Tab #1"></md-tab>
 *    <md-tab label="Tab #2"></md-tab>
 *    <md-tab label="Tab #3"></md-tab>
 *  <md-tabs>
 *  </hljs>
 *
 * Tabs supports three (3) usage scenarios:
 *
 *  1. Tabs (buttons only)
 *  2. Tabs with internal view content
 *  3. Tabs with external view content
 *
 * **Tab-only** support is useful when tab buttons are used for custom navigation regardless of any other components, content, or views.
 * **Tabs with internal views** are the traditional usages where each tab has associated view content and the view switching is managed internally by the Tabs component.
 * **Tabs with external view content** is often useful when content associated with each tab is independently managed and data-binding notifications announce tab selection changes.
 *
 * > As a performance bonus, if the tab content is managed internally then the non-active (non-visible) tab contents are temporarily disconnected from the `$scope.$digest()` processes; which restricts and optimizes DOM updates to only the currently active tab.
 *
 * Additional features also include:
 *
 * *  Content can include any markup.
 * *  If a tab is disabled while active/selected, then the next tab will be auto-selected.
 * *  If the currently active tab is the last tab, then next() action will select the first tab.
 * *  Any markup (other than **`<md-tab>`** tags) will be transcluded into the tab header area BEFORE the tab buttons.
 *
 * ### Explanation of tab stretching
 *
 * Initially, tabs will have an inherent size.  This size will either be defined by how much space is needed to accommodate their text or set by the user through CSS.  Calculations will be based on this size.
 *
 * On mobile devices, tabs will be expanded to fill the available horizontal space.  When this happens, all tabs will become the same size.
 *
 * On desktops, by default, stretching will never occur.
 *
 * This default behavior can be overridden through the `md-stretch-tabs` attribute.  Here is a table showing when stretching will occur:
 *
 * `md-stretch-tabs` | mobile    | desktop
 * ------------------|-----------|--------
 * `auto`            | stretched | ---
 * `always`          | stretched | stretched
 * `never`           | ---       | ---
 *
 * @param {integer=} md-selected Index of the active/selected tab
 * @param {boolean=} md-no-ink If present, disables ink ripple effects.
 * @param {boolean=} md-no-bar If present, disables the selection ink bar.
 * @param {string=}  md-align-tabs Attribute to indicate position of tab buttons: `bottom` or `top`; default is `top`
 * @param {string=} md-stretch-tabs Attribute to indicate whether or not to stretch tabs: `auto`, `always`, or `never`; default is `auto`
 *
 * @usage
 * <hljs lang="html">
 * <md-tabs md-selected="selectedIndex" >
 *   <img ng-src="img/angular.png" class="centered">
 *
 *   <md-tab
 *      ng-repeat="tab in tabs | orderBy:predicate:reversed"
 *      md-on-select="onTabSelected(tab)"
 *      md-on-deselect="announceDeselected(tab)"
 *      disabled="tab.disabled" >
 *
 *       <md-tab-label>
 *           {{tab.title}}
 *           <img src="img/removeTab.png"
 *                ng-click="removeTab(tab)"
 *                class="delete" >
 *       </md-tab-label>
 *
 *       {{tab.content}}
 *
 *   </md-tab>
 *
 * </md-tabs>
 * </hljs>
 *
 */
function TabsDirective($mdTheming) {
  return {
    restrict: 'E',
    controller: '$mdTabs',
    require: 'mdTabs',
    transclude: true,
    scope: {
      selectedIndex: '=?mdSelected'
    },
    template:
      '<section class="md-header" ' +
        'ng-class="{\'md-paginating\': pagination.active}">' +

        '<button class="md-paginator md-prev" ' +
          'ng-if="pagination.active && pagination.hasPrev" ' +
          'ng-click="pagination.clickPrevious()" ' +
          'aria-hidden="true">' +
        '</button>' +

        // overflow: hidden container when paginating
        '<div class="md-header-items-container" md-tabs-pagination>' +
          // flex container for <md-tab> elements
          '<div class="md-header-items">' +
            '<md-tabs-ink-bar></md-tabs-ink-bar>' +
          '</div>' +
        '</div>' +

        '<button class="md-paginator md-next" ' +
          'ng-if="pagination.active && pagination.hasNext" ' +
          'ng-click="pagination.clickNext()" ' +
          'aria-hidden="true">' +
        '</button>' +

      '</section>' +
      '<section class="md-tabs-content"></section>',
    link: postLink
  };

  function postLink(scope, element, attr, tabsCtrl, transclude) {

    scope.stretchTabs = attr.hasOwnProperty('mdStretchTabs') ? attr.mdStretchTabs || 'always' : 'auto';

    $mdTheming(element);
    configureAria();
    watchSelected();

    transclude(scope.$parent, function(clone) {
      angular.element(element[0].querySelector('.md-header-items')).append(clone);
    });

    function configureAria() {
      element.attr('role', 'tablist');
    }

    function watchSelected() {
      scope.$watch('selectedIndex', function watchSelectedIndex(newIndex, oldIndex) {
        if (oldIndex == newIndex) return;
        var rightToLeft = oldIndex > newIndex;
        tabsCtrl.deselect(tabsCtrl.itemAt(oldIndex), rightToLeft);

        if (tabsCtrl.inRange(newIndex)) {
          var newTab = tabsCtrl.itemAt(newIndex);
          while (newTab && newTab.isDisabled()) {
            newTab = newIndex > oldIndex 
                ? tabsCtrl.next(newTab)
                : tabsCtrl.previous(newTab);
          }
          tabsCtrl.select(newTab, rightToLeft);
        }
      });
    }
  }
}
TabsDirective.$inject = ["$mdTheming"];
})();

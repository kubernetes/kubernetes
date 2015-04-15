// https://github.com/kangax/detect-global

// tweaked to run without a UI.

(function () {
    function getPropertyDescriptors(object) {
      var props = { };
      for (var prop in object) {
        
        // nerfing for firefox who goes crazy over some objects like sessionStorage
        try {
          
          props[prop] = {
            type:  typeof object[prop],
            value: object[prop]
          };
          
        } catch(e){
          props[prop] = {}; 
        }
      }
      return props;
    }
    
    function getCleanWindow() {
      var elIframe = document.createElement('iframe');
      elIframe.style.display = 'none';
      
      var ref = document.getElementsByTagName('script')[0];
      ref.parentNode.insertBefore(elIframe, ref);
      
      elIframe.src = 'about:blank';
      return elIframe.contentWindow;
    }
    
    function appendControl(el, name) {
      var elCheckbox = document.createElement('input');
      elCheckbox.type = 'checkbox';
      elCheckbox.checked = true;
      elCheckbox.id = '__' + name;
      
      var elLabel = document.createElement('label');
      elLabel.htmlFor = '__' + name;
      elLabel.innerHTML = 'Exclude ' + name + ' properties?';
      elLabel.style.marginLeft = '0.5em';
      
      var elWrapper = document.createElement('p');
      elWrapper.style.marginBottom = '0.5em';
      
      elWrapper.appendChild(elCheckbox);
      elWrapper.appendChild(elLabel);

      el.appendChild(elWrapper);
    }
    
    function appendAnalyze(el) {
      var elAnalyze = document.createElement('button');
      elAnalyze.id = '__analyze';
      elAnalyze.innerHTML = 'Analyze';
      elAnalyze.style.marginTop = '1em';
      el.appendChild(elAnalyze);
    }
    
    function appendCancel(el) {
      var elCancel = document.createElement('a');
      elCancel.href = '#';
      elCancel.innerHTML = 'Cancel';
      elCancel.style.cssText = 'color:#eee;margin-left:0.5em;';
      elCancel.onclick = function() {
        el.parentNode.removeChild(el);
        return false; 
      };
      el.appendChild(elCancel);
    }
    
    function initConfigPopup() {
      var el = document.createElement('div');
      
      el.style.cssText =  'position:fixed; left:10px; top:10px; width:300px; background:rgba(50,50,50,0.9);' +
                          '-moz-border-radius:10px; padding:1em; color: #eee; text-align: left;' +
                          'font-family: "Helvetica Neue", Verdana, Arial, sans serif; z-index: 99999;';
      
      for (var prop in propSets) {
        appendControl(el, prop);
      }
      
      appendAnalyze(el);
      appendCancel(el);
      
      var ref = document.getElementsByTagName('script')[0];
      ref.parentNode.insertBefore(el, ref);
    }
    
    function getPropsCount(object) {
      var count = 0;
      for (var prop in object) {
        count++;
      }
      return count;
    }
    
    function shouldDeleteProperty(propToCheck) {
      for (var prop in propSets) {
        var elCheckbox = document.getElementById('__' + prop);
        var isPropInSet = propSets[prop].indexOf(propToCheck) > -1;
        if (isPropInSet && (elCheckbox ? elCheckbox.checked : true) ) {
          return true;
        }
      }
    }
    
    function analyze() {
      var global = (function(){ return this; })(),
          globalProps = getPropertyDescriptors(global),
          cleanWindow = getCleanWindow();
          
      for (var prop in cleanWindow) {
        if (globalProps[prop]) {
          delete globalProps[prop];
        }
      }
      for (var prop in globalProps) {
        if (shouldDeleteProperty(prop)) {
          delete globalProps[prop];
        }
      }
      
      window.__globalsCount = getPropsCount(globalProps);
      window.__globals      = globalProps;
      
      window.console && console.log('Total number of global properties: ' + __globalsCount);
      window.console && console.dir(__globals);
    }
    
    var propSets = {
      'Prototype':        '$$ $A $F $H $R $break $continue $w Abstract Ajax Class Enumerable Element Field Form ' +
                          'Hash Insertion ObjectRange PeriodicalExecuter Position Prototype Selector Template Toggle Try'.split(' '),
                        
      'Scriptaculous':    'Autocompleter Builder Control Draggable Draggables Droppables Effect Sortable SortableObserver Sound Scriptaculous'.split(' '),
      'Firebug':          'loadFirebugConsole console _getFirebugConsoleElement _FirebugConsole _FirebugCommandLine _firebug'.split(' '),
      'Mozilla':          'Components XPCNativeWrapper XPCSafeJSObjectWrapper getInterface netscape GetWeakReference GeckoActiveXObject'.split(' '),
      'GoogleAnalytics':  'gaJsHost gaGlobal _gat _gaq pageTracker'.split(' '),
      'lazyGlobals':      'onhashchange'.split(' ')
    };
    
    // initConfigPopup(); // disable because we're going UI-less.
    
    var analyzeElem = document.getElementById('__analyze');
    analyzeElem && (analyzeElem.onclick = analyze);
    
    analyze(); // and assign total added globals to window.__globalsCount
    
})();
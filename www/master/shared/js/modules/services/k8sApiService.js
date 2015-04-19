app.provider('k8sApi',
             function() {

               var urlBase = '';

               this.setUrlBase = function(value) { urlBase = value; };

               var _get = function($http, baseUrl, query) {
                 var _fullUrl = baseUrl;
                 if (query !== undefined) {
                   _fullUrl += '/' + query;
                 }

                 return $http.get(_fullUrl);
               };

               this.$get = function($http, $q) {
                 var api = {};

                 api.getUrlBase = function() { return urlBase; };

                 api.getPods = function(query) { return _get($http, urlBase + '/pods', query); };

                 api.getMinions = function(query) { return _get($http, urlBase + '/minions', query); };

                 api.getServices = function(query) { return _get($http, urlBase + '/services', query); };

                 api.getReplicationControllers = function(query) {
                   return _get($http, urlBase + '/replicationControllers', query)
                 };

                 api.getEvents = function(query) { return _get($http, urlBase + '/events', query); };

                 return api;
               };
             })
    .config(function(k8sApiProvider, ENV) {
      if (ENV && ENV['/'] && ENV['/']['k8sApiServer']) {
        var proxy = ENV['/']['cAdvisorProxy'] || '';
        k8sApiProvider.setUrlBase(proxy + ENV['/']['k8sApiServer']);
      }
    });

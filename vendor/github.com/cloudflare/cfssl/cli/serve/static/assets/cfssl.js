(function() {
  'use strict';
  /* globals m */

  // > framework extensions
  m.deferred.resolve = function (value) {
    var deferred = m.deferred();
    deferred.resolve(value);
    return deferred.promise;
  };

  m.deferred.reject = function (value) {
    var deferred = m.deferred();
    deferred.reject(value);
    return deferred.promise;
  };
  // < framework extensions

  var page = (function() {
    var title = '';

    return {
      title: function(store) {
        if (arguments.length > 0) {
          title = store;

          if (!title.length) {
            document.title = 'CFSSL';
          } else {
            document.title = title + ' | CFSSL';
          }
        }

        return title;
      }
    };
  }());

  // > i18n support
  var phrases = {
    'bundle.title': 'Bundle',
    'bundle.action': 'Bundle',
    'bundle.flavor': 'Flavor',
    'bundle.ubiquitous': 'Ubiquitous',
    'bundle.optimal': 'Optimal',
    'bundle.force': 'Force',
    'bundle.bundle.title': 'Bundled',
    'common.server': 'Server',
    'common.cipher': 'Cipher',
    'common.packages': 'Packages',
    'common.host': 'Host',
    'scan.title': 'Scan',
    'scan.action': 'Scan',
    'scan.broad.title': 'Broad',
    'scan.broad.description': 'Large scale scans of TLS hosts.',
    'scan.broad.ICAs.title': 'Intermediate Certificate Authorities',
    'scan.broad.ICAs.body': 'Scans a CIDR IP range for unknown Intermediate Certificate Authorities.',
    'scan.connectivity.title': 'Connectivity',
    'scan.connectivity.description': 'Scans for basic connectivity with the host through DNS and TCP/TLS dials.',
    'scan.connectivity.DNSLookup.title': 'DNS Lookup',
    'scan.connectivity.DNSLookup.body': 'Determines if the host can be resolved through DNS.',
    'scan.connectivity.TCPDial.title': 'TCP Dial',
    'scan.connectivity.TCPDial.body': 'Determines if host accepts TCP connection.',
    'scan.connectivity.TLSDial.title': 'TLS Dial',
    'scan.connectivity.TLSDial.body': 'Tests if host can perform TLS handshake.',
    'scan.pki.title': 'Public-Key Infrastructure',
    'scan.pki.description': 'Scans for the Public-Key Infrastructure.',
    'scan.pki.ChainExpiration.title': 'Chain Expiration',
    'scan.pki.ChainExpiration.body': 'Ensures host\'s chain hasn\'t expired and won\'t expire in the next 30 days.',
    'scan.pki.ChainValidation.title': 'Chain Validation',
    'scan.pki.ChainValidation.body': 'Looks at all certificate in the host\'s chain, and ensures they are all valid.',
    'scan.tlshandshake.title': 'TLS Handshake',
    'scan.tlshandshake.description': 'Scans for host\'s SSL and TLS versions and cipher suite negotiation.',
    'scan.tlshandshake.CipherSuite.title': 'Cipher Suite Matrix',
    'scan.tlshandshake.CipherSuite.body': 'Determines host\'s cipher suite accepted and prefered order.',
    'scan.tlssession.title': 'TLS Session',
    'scan.tlssession.description': 'Scans host\'s implementation of TLS session resumption using session ticket and session IDs.',
    'scan.tlssession.SessionResume.title': 'Session Resumption',
    'scan.tlssession.SessionResume.body': 'Confirms the host is able to resume sessions across all addresses.',
    'scan.tlssession.SessionResume.supports_session_resume': 'Supports Session Resumption'
  };

  // stub to replace with intl-messageformat
  function Tformat(key) {
    return phrases[key] || '';
  }

  // < i18n support

  function appWrapper(module) {
    function navLink(selector, route, name) {
      var isActive = m.route().indexOf(route) === 0;
      selector += '[href="' + route + '"]';

      return m('li' + (isActive ? '.active' : ''), [
        m(selector, {
          config: m.route
        }, name)
      ]);
    }
    return [
      m('nav.navbar.navbar-default.navbar-static-top', [
        m('.container', [
          m('.navbar-header', [
            m('a.navbar-brand[href="/"]', {
              config: m.route
            }, 'CFSSL')
          ]),
          m('.collapse.navbar-collapse', [
            m('ul.nav.navbar-nav', [
              navLink('a', '/scan', Tformat('scan.title')),
              navLink('a', '/bundle', Tformat('bundle.title'))
            ]),
            m('ul.nav.navbar-nav.navbar-right', [
              m('li', m('a[href="https://pkg.cfssl.org"]', Tformat('common.packages'))),
              m('li', m('a[href="https://github.com/cloudflare/cfssl"]', 'GitHub')),
            ])
          ])
        ])
      ]),
      m('.container', module),
      m('footer.container', {
        style: {
          paddingTop: '40px',
          paddingBottom: '40px',
          marginTop: '100px',
          borderTop: '1px solid #e5e5e5',
          textAlign: 'center'
        }
      }, [
        m('p', [
          'Code licensed under ',
          m('a[href="https://github.com/cloudflare/cfssl/blob/master/LICENSE"]', 'BSD-2-Clause'),
          '.'
        ])
      ])
    ];
  }

  var panel = {
    view: function(ctrl, args, children) {
      function gradeToGlyphicon(grade) {
        switch(grade) {
          case 'Good':
            return 'glyphicon-ok-sign';
          case 'Warning':
            return 'glyphicon-exclamation-sign';
          case 'Bad':
            return 'glyphicon-remove-sign';
          default:
            return 'glyphicon-question-sign';
        }
      }

      function gradeToPanel(grade) {
        switch(grade) {
          case 'Good':
            return 'panel-success';
          case 'Warning':
            return 'panel-warning';
          case 'Bad':
            return 'panel-danger';
          default:
            return 'panel-default';
        }
      }

      if (!args.grade) {
        return m('.panel.panel-default', [
          m('.panel-heading', args.title),
          m('.panel-body', args.body),
          children
        ])
      }

      return m('.panel.' + gradeToPanel(args.grade), [
        m('.panel-heading', [
          m('span.glyphicon.' + gradeToGlyphicon(args.grade)),
          ' ',
          args.title
        ]),
        m('.panel-body', args.body),
        children
      ]);
    }
  };

  var table = {
    view: function(ctrl, args) {
      return m('table.table.table-bordered.table-striped', [
        m('thead', [
          m('tr', args.columns.map(function(column) {
            return m('th', column);
          }))
        ]),
        m('tbody', args.rows.map(function(row) {
          return m('tr', row.map(function(cell) {
            return m('td', cell);
          }));
        }))
      ]);
    }
  };

  var listGroup = {
    view: function(ctrl, children) {
      return m('ul.list-group', children.map(function(item) {
        return m('li.list-group-item', item);
      }));
    }
  };

  var home = {
    controller: function() {
      page.title('');
      return;
    },
    view: function() {
      return appWrapper([
        m('h1.page-header', 'CFSSL: CloudFlare\'s PKI toolkit'), m('p', [
          'See ',
          m('a[href="https://blog.cloudflare.com/introducing-cfssl"]', 'blog post'),
          ' or ',
          m('a[href="https://github.com/cloudflare/cfssl"]', 'contribute on GitHub'),
          '.'
        ])
      ]);
    }
  };

  var scan = {
    vm: {
      init: function(domain) {
        scan.vm.domain = m.prop(domain ? domain : '');
        scan.vm.loading = m.prop(false);
        scan.vm.Scan = m.prop(false);
        scan.vm.scan = function(evt) {
          var domain = scan.vm.domain();
          scan.vm.Scan(false);
          scan.vm.loading(true);

          if (evt) {
            evt.preventDefault();
          }

          setTimeout(function() {
            scan.Model.scan(domain).then(function(result) {
              scan.vm.loading(false);
              scan.vm.Scan(result);
            });
          }, 0);
        };

        // TODO: remove!
        if (domain) {
          scan.vm.loading(true);
          setTimeout(function() {
            scan.Model.scan(domain).then(function(result) {
              scan.vm.loading(false);
              scan.vm.Scan(result);
            });
          }, 0);
        }
      }
    },
    Model: function(data) {
      this.domain = m.prop(data.domain);
      this.IntermediateCAs = m.prop(data.IntermediateCAs);
      this.DNSLookup = m.prop(data.DNSLookup);
      this.TCPDial = m.prop(data.TCPDial);
      this.TLSDial = m.prop(data.TLSDial);
      this.ChainExpiration = m.prop(data.ChainExpiration);
      this.ChainValidation = m.prop(data.ChainValidation);
      this.CipherSuite = m.prop(data.CipherSuite);
      this.SessionResume = m.prop(data.SessionResume);
    },
    controller: function() {
      scan.vm.init(m.route.param('domain'));
      page.title(Tformat('scan.title'))
      return;
    },
    view: function() {
      function broad() {
        var ICAs = results.IntermediateCAs();
        var out = [];

        out.push(m('h3.page-header', Tformat('scan.broad.title')), m('p', Tformat('scan.broad.description')));

        if (ICAs && ICAs.grade) {
          out.push(m.component(panel, {
            grade: ICAs.grade,
            title: Tformat('scan.broad.ICAs.title'),
            body: Tformat('scan.broad.ICAs.body')
          }));
        }

        if (out.length === 2) {
          return;
        }

        return out;
      }

      function connectivity() {
        var DNSLookup = results.DNSLookup();
        var TCPDial = results.TCPDial();
        var TLSDial = results.TLSDial();
        var out = [];

        out.push(m('h3.page-header', Tformat('scan.connectivity.title')), m('p', Tformat('scan.connectivity.description')));

        if (DNSLookup && DNSLookup.grade) {
          out.push(m.component(panel, {
            grade: DNSLookup.grade,
            title: Tformat('scan.connectivity.DNSLookup.title'),
            body: Tformat('scan.connectivity.DNSLookup.body')
          }, m.component(listGroup, DNSLookup.output.sort())));
        }

        if (TCPDial && TCPDial.grade) {
          out.push(m.component(panel, {
            grade: TCPDial.grade,
            title: Tformat('scan.connectivity.TCPDial.title'),
            body: Tformat('scan.connectivity.TCPDial.body')
          }));
        }

        if (TLSDial && TLSDial.grade) {
          out.push(m.component(panel, {
            grade: TLSDial.grade,
            title: Tformat('scan.connectivity.TLSDial.title'),
            body: Tformat('scan.connectivity.TLSDial.body')
          }));
        }

        if (out.length === 2) {
          return;
        }

        return out;
      }

      function tlssession() {
        var SessionResume = results.SessionResume();
        var out = [];
        var body;

        out.push(m('h3.page-header', Tformat('scan.tlssession.title')), m('p', Tformat('scan.tlssession.description')));

        if (SessionResume && SessionResume.grade) {
          body = null;
          if (SessionResume.output) {
            body = m.component(table, {
              columns: [Tformat('common.server'), Tformat('scan.tlssession.SessionResume.supports_session_resume')],
              rows: Object.keys(SessionResume.output).sort().map(function(ip) {
                var supported = SessionResume.output[ip];

                return [
                  ip,
                  m('span.glyphicon.glyphicon-' + (supported ? 'ok-sign' : 'remove-sign'))
                ];
              })
            });
          }

          out.push(m.component(panel, {
            grade: SessionResume.grade,
            title: Tformat('scan.tlssession.SessionResume.title'),
            body: Tformat('scan.tlssession.SessionResume.body')
          }, body));
        }

        if (out.length === 2) {
          return;
        }

        return out;
      }

      function pki() {
        var ChainExpiration = results.ChainExpiration();
        var ChainValidation = results.ChainValidation();
        var out = [];
        var body;

        out.push(m('h3.page-header', Tformat('scan.pki.title')), m('p', Tformat('scan.pki.description')));

        if (ChainExpiration && ChainExpiration.grade) {
          body = null;
          if (ChainExpiration.output) {
            body = m.component(listGroup, [
              m('time[datetime="' + ChainExpiration.output + '"]', (new Date(ChainExpiration.output)).toLocaleString('bestfit', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: 'numeric',
                minute: 'numeric',
                second: 'numeric',
                timeZone: 'UTC',
                timeZoneName: 'short'
              }))
            ]);
          }

          out.push(m.component(panel, {
            grade: ChainExpiration.grade,
            title: Tformat('scan.pki.ChainExpiration.title'),
            body: Tformat('scan.pki.ChainExpiration.body')
          }, body));
        }

        if (ChainValidation && ChainValidation.grade) {
          body = null;
          if (ChainValidation.output && Array.isArray(ChainValidation.output)) {
            body = m.component(listGroup, ChainValidation.output);
          }

          out.push(m.component(panel, {
            grade: ChainValidation.grade,
            title: Tformat('scan.pki.ChainValidation.title'),
            body: Tformat('scan.pki.ChainValidation.body')
          }, body));
        }

        if (out.length === 2) {
          return
        }

        return out;
      }

      function tlshandshake() {
        var CipherSuite = results.CipherSuite();
        var out = [];
        var body;

        out.push(m('h3.page-header', Tformat('scan.tlshandshake.title')), m('p', Tformat('scan.tlshandshake.description')));

        if (CipherSuite && CipherSuite.grade) {
          body = null;
          if (CipherSuite.output) {
            body = m.component(table, {
              columns: [Tformat('common.cipher'), 'TLS 1.2', 'TLS 1.1', 'TLS 1.0', 'SSL 3.0'],
              rows: CipherSuite.output.map(function(results) {
                var cipher = Object.keys(results)[0];
                var result = results[cipher];

                if (typeof result[0] === 'string') {
                  return [
                    cipher,
                    result.indexOf('TLS 1.2') !== -1 ? m('span.glyphicon.glyphicon-ok-sign') : '-',
                    result.indexOf('TLS 1.1') !== -1 ? m('span.glyphicon.glyphicon-ok-sign') : '-',
                    result.indexOf('TLS 1.0') !== -1 ? m('span.glyphicon.glyphicon-ok-sign') : '-',
                    result.indexOf('SSL 3.0') !== -1 ? m('span.glyphicon.glyphicon-remove-sign') : '-'
                  ];
                }

                return [
                  cipher,
                  result[0] && result[0]['TLS 1.2'][0] || '-',
                  result[1] && result[1]['TLS 1.1'][0] || '-',
                  result[2] && result[2]['TLS 1.0'][0] || '-',
                  result[3] && result[3]['SSL 3.0'][0] || '-',
                ];
              })
            });
          }

          out.push(m.component(panel, {
            grade: CipherSuite.grade,
            title: Tformat('scan.tlshandshake.CipherSuite.title'),
            body: Tformat('scan.tlshandshake.CipherSuite.body')
          }, body));
        }

        if (out.length === 2) {
          return
        }

        return out;
      }

      var results = scan.vm.Scan();
      return appWrapper([
        m('h1.page-header', Tformat('scan.title')),
        m('form.form-horizontal', [
          m('.form-group', [
            m('label.col-sm-2.control-label[for=scanhost]', Tformat('common.host')),
            m('.col-sm-8', [
              m('input.form-control#scanhost[placeholder="cfssl.org"]', {
                value: scan.vm.domain(),
                onchange: m.withAttr('value', scan.vm.domain)
              })
            ])
          ]),
          m('.form-group', [
            m('.col-sm-offset-2 col-sm-10', [
              m('button.btn.btn-default[type="submit"]', {
                onclick: scan.vm.scan,
                disabled: scan.vm.loading()
              }, Tformat('scan.action'))
            ])
          ])
        ]),
        !scan.vm.loading() ? '' : [
          m('p', 'Scanning ' + scan.vm.domain())
        ],
        !results ? '' : [
          m('h2.page-header', 'Results for ' + scan.vm.Scan().domain()),
          broad(),
          connectivity(),
          tlssession(),
          pki(),
          tlshandshake()
        ]
      ]);
    }
  };

  scan.Model.scan = function(domain) {
    if (domain) {
      return m.request({
        method: 'GET',
        url: '/api/v1/cfssl/scan',
        data: {
          host: domain
        },
        unwrapSuccess: function(response) {
          if (!response.success) {
            throw new Error(response.messages.join(', '));
          }

          return response.result;
        },
        unwrapError: function(response) {
          return response.errors;
        }
      })
      .then(function(response) {
        var results = new scan.Model({
          domain: domain,
          IntermediateCAs: response.Broad && response.Broad.IntermediateCAs,
          DNSLookup: response.Connectivity.DNSLookup,
          TCPDial: response.Connectivity.TCPDial,
          TLSDial: response.Connectivity.TLSDial,
          ChainExpiration: response.PKI.ChainExpiration,
          ChainValidation: response.PKI.ChainValidation,
          CipherSuite: response.TLSHandshake.CipherSuite,
          SessionResume: response.TLSSession.SessionResume
        });

        return results;
      });
    }

    return m.deferred.reject();
  };

  var bundle = {
    vm: {
      init: function(domain) {
        bundle.vm.domain = m.prop(domain ? domain : '');
        bundle.vm.flavor = m.prop('ubiquitous');
        bundle.vm.loading = m.prop(false);
        bundle.vm.Bundle = m.prop(false);
        bundle.vm.bundle = function(evt) {
          var domain = bundle.vm.domain();
          var flavor = bundle.vm.flavor();
          bundle.vm.Bundle(false);
          bundle.vm.loading(true);

          if (evt) {
            evt.preventDefault();
          }

          setTimeout(function() {
            bundle.Model.bundle(domain, flavor).then(function(result) {
              bundle.vm.loading(false);
              bundle.vm.Bundle(result);
            });
          }, 0);
        };

        // TODO: remove!
        if (domain) {
          bundle.vm.loading(true);
          setTimeout(function() {
            bundle.Model.bundle(domain, bundle.vm.flavor()).then(function(result) {
              bundle.vm.loading(false);
              bundle.vm.Bundle(result);
            });
          }, 0);
        }
      }
    },
    Model: function(data) {
      this.domain = m.prop(data.domain);
      this.bundle = m.prop(data.bundle);
      this.expires = m.prop(data.expires);
      this.messages = m.prop(data.messages);
      this.oscp = m.prop(data.oscp);
    },
    controller: function() {
      bundle.vm.init(m.route.param('domain'));
      page.title(Tformat('bundle.title'))
      return;
    },
    view: function() {
      var results = bundle.vm.Bundle();
      return appWrapper([
        m('h1.page-header', Tformat('bundle.title')),
        m('form.form-horizontal', [
          m('.form-group', [
            m('label.col-sm-2.control-label[for=bundlehost]', Tformat('common.host')),
            m('.col-sm-8', [
              m('input.form-control#bundlehost[placeholder="cfssl.org"]', {
                value: bundle.vm.domain(),
                onchange: m.withAttr('value', bundle.vm.domain)
              })
            ])
          ]),
          m('.form-group', [
            m('label.col-sm-2.control-label[for=bundleflavor]', Tformat('bundle.flavor')),
            m('.col-sm-8', [
              m('select#bundleflavor', {
                value: bundle.vm.flavor(),
                onchange: m.withAttr('value', bundle.vm.flavor)
              }, [
                m('option[value="ubiquitous"]', Tformat('bundle.ubiquitous')),
                m('option[value="optimal"]', Tformat('bundle.optimal')),
                m('option[value="force"]', Tformat('bundle.force'))
              ])
            ])
          ]),
          m('.form-group', [
            m('.col-sm-offset-2 col-sm-10', [
              m('button.btn.btn-default[type="submit"]', {
                onclick: bundle.vm.bundle,
                disabled: bundle.vm.loading()
              }, Tformat('bundle.action'))
            ])
          ])
        ]),
        !bundle.vm.loading() ? '' : [
          m('p', 'Bundling ' + bundle.vm.domain())
        ],
        !results ? '' : [
          m('h2.page-header', 'Results for ' + bundle.vm.Bundle().domain()),
          m.component(panel, {
            title: Tformat('bundle.bundle.title'),
            body: m('pre', results.bundle())
          }, !results.messages().length ? '' : m.component(listGroup, results.messages())),
        ]
      ]);
    }
  };

  bundle.Model.bundle = function(domain, flavor) {
    if (domain && flavor) {
      return m.request({
        method: 'POST',
        url: '/api/v1/cfssl/bundle',
        data: {
          domain: domain,
          flavor: flavor
        },
        unwrapSuccess: function(response) {
          if (!response.success) {
            throw new Error(response.messages.join(', '));
          }

          return response.result;
        },
        unwrapError: function(response) {
          return response.errors;
        }
      })
      .then(function(response) {
        var results = new bundle.Model({
          domain: domain,
          bundle: response.bundle,
          expires: response.expires,
          messages: response.status && response.status.messages || [],
          oscp: response.oscp_support
        });

        return results;
      });
    }

    return m.deferred.reject();
  }

  m.route.mode = 'pathname';

  m.route(document.body, '/', {
    '/': home,
    '/bundle': bundle,
    '/bundle/:domain': bundle,
    '/scan': scan,
    '/scan/:domain': scan
  });

  window.scan = scan;
  window.bundle = bundle;
}());

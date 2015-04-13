

// XML HTTP Request Level 2
// www.w3.org/TR/XMLHttpRequest2/

// Much more details at github.com/Modernizr/Modernizr/issues/385

// all three of these details report consistently across all target browsers:
//   !!(window.ProgressEvent);
//   !!(window.FormData);
//   window.XMLHttpRequest && "withCredentials" in new XMLHttpRequest;

Modernizr.addTest('xhr2', 'FormData' in window);

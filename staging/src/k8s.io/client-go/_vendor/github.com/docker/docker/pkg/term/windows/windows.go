// These files implement ANSI-aware input and output streams for use by the Docker Windows client.
// When asked for the set of standard streams (e.g., stdin, stdout, stderr), the code will create
// and return pseudo-streams that convert ANSI sequences to / from Windows Console API calls.

package windows

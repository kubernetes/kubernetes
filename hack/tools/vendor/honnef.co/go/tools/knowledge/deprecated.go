package knowledge

const (
	DeprecatedNeverUse    = -1
	DeprecatedUseNoLonger = -2
)

type Deprecation struct {
	DeprecatedSince           int
	AlternativeAvailableSince int
}

// go/importer.ForCompiler contains "Deprecated:", but it refers to a single argument, not the whole function.
// Luckily, the notice starts in the middle of a paragraph, and as such isn't detected by us.

var StdlibDeprecations = map[string]Deprecation{
	// FIXME(dh): AllowBinary isn't being detected as deprecated
	// because the comment has a newline right after "Deprecated:"
	"go/build.AllowBinary":                      {7, 7},
	"(archive/zip.FileHeader).CompressedSize":   {1, 1},
	"(archive/zip.FileHeader).UncompressedSize": {1, 1},
	"(archive/zip.FileHeader).ModifiedTime":     {10, 10},
	"(archive/zip.FileHeader).ModifiedDate":     {10, 10},
	"(*archive/zip.FileHeader).ModTime":         {10, 10},
	"(*archive/zip.FileHeader).SetModTime":      {10, 10},
	"(go/doc.Package).Bugs":                     {1, 1},
	"os.SEEK_SET":                               {7, 7},
	"os.SEEK_CUR":                               {7, 7},
	"os.SEEK_END":                               {7, 7},
	"(net.Dialer).Cancel":                       {7, 7},
	"runtime.CPUProfile":                        {9, 0},
	"compress/flate.ReadError":                  {6, DeprecatedUseNoLonger},
	"compress/flate.WriteError":                 {6, DeprecatedUseNoLonger},
	"path/filepath.HasPrefix":                   {0, DeprecatedNeverUse},
	"(net/http.Transport).Dial":                 {7, 7},
	"(net/http.Transport).DialTLS":              {14, 14},
	"(*net/http.Transport).CancelRequest":       {6, 5},
	"net/http.ErrWriteAfterFlush":               {7, DeprecatedUseNoLonger},
	"net/http.ErrHeaderTooLong":                 {8, DeprecatedUseNoLonger},
	"net/http.ErrShortBody":                     {8, DeprecatedUseNoLonger},
	"net/http.ErrMissingContentLength":          {8, DeprecatedUseNoLonger},
	"net/http/httputil.ErrPersistEOF":           {0, DeprecatedUseNoLonger},
	"net/http/httputil.ErrClosed":               {0, DeprecatedUseNoLonger},
	"net/http/httputil.ErrPipeline":             {0, DeprecatedUseNoLonger},
	"net/http/httputil.ServerConn":              {0, 0},
	"net/http/httputil.NewServerConn":           {0, 0},
	"net/http/httputil.ClientConn":              {0, 0},
	"net/http/httputil.NewClientConn":           {0, 0},
	"net/http/httputil.NewProxyClientConn":      {0, 0},
	"(net/http.Request).Cancel":                 {7, 7},
	"(text/template/parse.PipeNode).Line":       {1, DeprecatedUseNoLonger},
	"(text/template/parse.ActionNode).Line":     {1, DeprecatedUseNoLonger},
	"(text/template/parse.BranchNode).Line":     {1, DeprecatedUseNoLonger},
	"(text/template/parse.TemplateNode).Line":   {1, DeprecatedUseNoLonger},
	"database/sql/driver.ColumnConverter":       {9, 9},
	"database/sql/driver.Execer":                {8, 8},
	"database/sql/driver.Queryer":               {8, 8},
	"(database/sql/driver.Conn).Begin":          {8, 8},
	"(database/sql/driver.Stmt).Exec":           {8, 8},
	"(database/sql/driver.Stmt).Query":          {8, 8},
	"syscall.StringByteSlice":                   {1, 1},
	"syscall.StringBytePtr":                     {1, 1},
	"syscall.StringSlicePtr":                    {1, 1},
	"syscall.StringToUTF16":                     {1, 1},
	"syscall.StringToUTF16Ptr":                  {1, 1},
	"(*regexp.Regexp).Copy":                     {12, DeprecatedUseNoLonger},
	"(archive/tar.Header).Xattrs":               {10, 10},
	"archive/tar.TypeRegA":                      {11, 1},
	"go/types.NewInterface":                     {11, 11},
	"(*go/types.Interface).Embedded":            {11, 11},
	"go/importer.For":                           {12, 12},
	"encoding/json.InvalidUTF8Error":            {2, DeprecatedUseNoLonger},
	"encoding/json.UnmarshalFieldError":         {2, DeprecatedUseNoLonger},
	"encoding/csv.ErrTrailingComma":             {2, DeprecatedUseNoLonger},
	"(encoding/csv.Reader).TrailingComma":       {2, DeprecatedUseNoLonger},
	"(net.Dialer).DualStack":                    {12, 12},
	"net/http.ErrUnexpectedTrailer":             {12, DeprecatedUseNoLonger},
	"net/http.CloseNotifier":                    {11, 7},
	// This is hairy. The notice says "Not all errors in the http package related to protocol errors are of type ProtocolError", but doesn't that imply that some errors do?
	"net/http.ProtocolError":                      {8, DeprecatedUseNoLonger},
	"(crypto/x509.CertificateRequest).Attributes": {5, 3},

	// These functions have no direct alternative, but they are insecure and should no longer be used.
	"crypto/x509.IsEncryptedPEMBlock": {16, DeprecatedNeverUse},
	"crypto/x509.DecryptPEMBlock":     {16, DeprecatedNeverUse},
	"crypto/x509.EncryptPEMBlock":     {16, DeprecatedNeverUse},
	"crypto/dsa":                      {16, DeprecatedNeverUse},

	// This function has no alternative, but also no purpose.
	"(*crypto/rc4.Cipher).Reset":                     {12, DeprecatedNeverUse},
	"(net/http/httptest.ResponseRecorder).HeaderMap": {11, 7},
	"image.ZP":                                    {13, 0},
	"image.ZR":                                    {13, 0},
	"(*debug/gosym.LineTable).LineToPC":           {2, 2},
	"(*debug/gosym.LineTable).PCToLine":           {2, 2},
	"crypto/tls.VersionSSL30":                     {13, DeprecatedNeverUse},
	"(crypto/tls.Config).NameToCertificate":       {14, DeprecatedUseNoLonger},
	"(*crypto/tls.Config).BuildNameToCertificate": {14, DeprecatedUseNoLonger},
	"(crypto/tls.Config).SessionTicketKey":        {16, 5},
	// No alternative, no use
	"(crypto/tls.ConnectionState).NegotiatedProtocolIsMutual": {16, DeprecatedNeverUse},
	// No alternative, but insecure
	"(crypto/tls.ConnectionState).TLSUnique": {16, DeprecatedNeverUse},
	"image/jpeg.Reader":                      {4, DeprecatedNeverUse},

	// All of these have been deprecated in favour of external libraries
	"syscall.AttachLsf":                     {7, 0},
	"syscall.DetachLsf":                     {7, 0},
	"syscall.LsfSocket":                     {7, 0},
	"syscall.SetLsfPromisc":                 {7, 0},
	"syscall.LsfJump":                       {7, 0},
	"syscall.LsfStmt":                       {7, 0},
	"syscall.BpfStmt":                       {7, 0},
	"syscall.BpfJump":                       {7, 0},
	"syscall.BpfBuflen":                     {7, 0},
	"syscall.SetBpfBuflen":                  {7, 0},
	"syscall.BpfDatalink":                   {7, 0},
	"syscall.SetBpfDatalink":                {7, 0},
	"syscall.SetBpfPromisc":                 {7, 0},
	"syscall.FlushBpf":                      {7, 0},
	"syscall.BpfInterface":                  {7, 0},
	"syscall.SetBpfInterface":               {7, 0},
	"syscall.BpfTimeout":                    {7, 0},
	"syscall.SetBpfTimeout":                 {7, 0},
	"syscall.BpfStats":                      {7, 0},
	"syscall.SetBpfImmediate":               {7, 0},
	"syscall.SetBpf":                        {7, 0},
	"syscall.CheckBpfVersion":               {7, 0},
	"syscall.BpfHeadercmpl":                 {7, 0},
	"syscall.SetBpfHeadercmpl":              {7, 0},
	"syscall.RouteRIB":                      {8, 0},
	"syscall.RoutingMessage":                {8, 0},
	"syscall.RouteMessage":                  {8, 0},
	"syscall.InterfaceMessage":              {8, 0},
	"syscall.InterfaceAddrMessage":          {8, 0},
	"syscall.ParseRoutingMessage":           {8, 0},
	"syscall.ParseRoutingSockaddr":          {8, 0},
	"syscall.InterfaceAnnounceMessage":      {7, 0},
	"syscall.InterfaceMulticastAddrMessage": {7, 0},
	"syscall.FormatMessage":                 {5, 0},
	"syscall.PostQueuedCompletionStatus":    {17, 0},
	"syscall.GetQueuedCompletionStatus":     {17, 0},
	"syscall.CreateIoCompletionPort":        {17, 0},

	// Not marked as deprecated with a recognizable header, but deprecated nonetheless.
	"io/ioutil": {16, 16},
}

// Last imported from Go at 32b73ae18026e8a9dc4c5aa49999b1ea445bc68c with the following numbers of deprecations:
//
// chulak go@master ./src $ rg -c "Deprecated: "
// vendor/golang.org/x/crypto/curve25519/curve25519.go:1
// vendor/golang.org/x/text/transform/transform.go:1
// cmd/compile/internal/types/sym.go:2
// syscall/route_netbsd.go:1
// syscall/route_bsd.go:7
// syscall/lsf_linux.go:6
// syscall/exec_unix.go:1
// syscall/route_darwin.go:1
// syscall/route_freebsd.go:2
// syscall/route_dragonfly.go:2
// syscall/bpf_darwin.go:18
// syscall/route_openbsd.go:1
// syscall/syscall_windows.go:6
// syscall/syscall.go:3
// syscall/bpf_bsd.go:18
// cmd/compile/internal/types2/type.go:2
// syscall/exec_plan9.go:1
// cmd/internal/obj/textflag.go:1
// cmd/internal/obj/link.go:5
// cmd/vendor/golang.org/x/sys/unix/zsysnum_darwin_arm64.go:1
// cmd/vendor/golang.org/x/mod/semver/semver.go:1
// cmd/vendor/golang.org/x/sys/windows/syscall_windows.go:2
// cmd/vendor/golang.org/x/sys/unix/zsysnum_darwin_amd64.go:1
// cmd/vendor/golang.org/x/mod/modfile/rule.go:2
// cmd/vendor/golang.org/x/sys/windows/security_windows.go:1
// cmd/go/internal/modcmd/edit.go:1
// cmd/go/testdata/mod/example.com_deprecated_a_v1.9.0.txt:2
// cmd/go/testdata/mod/example.com_undeprecated_v1.0.0.txt:2
// cmd/go/testdata/mod/example.com_deprecated_b_v1.9.0.txt:2
// encoding/csv/reader.go:2
// encoding/json/decode.go:1
// encoding/json/encode.go:1
// cmd/go/testdata/script/mod_list_deprecated.txt:2
// cmd/go/testdata/script/mod_deprecate_message.txt:4
// cmd/go/testdata/script/mod_list_deprecated_replace.txt:1
// runtime/cpuprof.go:1
// cmd/go/testdata/script/mod_edit.txt:1
// crypto/tls/common.go:6
// crypto/rc4/rc4.go:1
// crypto/dsa/dsa.go:1
// path/filepath/path_unix.go:1
// path/filepath/path_windows.go:1
// path/filepath/path_plan9.go:1
// regexp/regexp.go:1
// crypto/x509/pem_decrypt.go:3
// crypto/x509/x509.go:1
// archive/zip/struct.go:6
// archive/tar/common.go:2
// debug/gosym/pclntab.go:2
// compress/flate/inflate.go:2
// image/geom.go:2
// image/jpeg/reader.go:1
// os/file.go:1
// net/dial.go:2
// net/http/server.go:2
// net/http/socks_bundle.go:1
// net/http/httputil/persist.go:8
// net/http/request.go:6
// net/http/transport.go:3
// net/http/httptest/recorder.go:1
// go/doc/doc.go:1
// go/types/errorcodes.go:1
// go/types/type.go:2
// database/sql/driver/driver.go:6
// text/template/parse/node.go:5
// go/importer/importer.go:2

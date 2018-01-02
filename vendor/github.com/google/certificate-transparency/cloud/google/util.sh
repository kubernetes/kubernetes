
function Header() {
  echo
  tput bold
  tput setaf 5
  echo $*
  tput sgr0
}

function WaitForStatus() {
  TYPE=$1
  NAME=$2
  ZONE=$3
  WANT_STATUS=$4
  STATUS=""
  until [[ "${STATUS}" =~ "${WANT_STATUS}" ]]; do
    sleep 1
    STATUS=$(${GCLOUD} compute ${TYPE} describe ${NAME} --zone ${ZONE} | grep "status:")
    echo ${STATUS}
  done
}

function WaitMachineUp() {
  INSTANCE=${1}
  ZONE=${2}
  echo "Waiting for ${INSTANCE}"
  until ${GCLOUD} compute ssh ${INSTANCE} --zone ${ZONE} --command "exit"; do
    sleep 1
    echo -n .
  done
  echo "${1} is up."
}

function WaitHttpStatus() {
  INSTANCE=${1}
  ZONE=${2}
  HTTP_PATH=${3}
  WANTED_STATUS=${4:-200}
  URL=${INSTANCE}:80${HTTP_PATH}
  echo "Waiting for HTTP ${WANTED_STATUS} from ${URL} "
  ${GCLOUD} compute ssh ${INSTANCE} \
    --zone ${ZONE} \
    --command "
      STATUS_CODE=''
      while [ \"\${STATUS_CODE}\" != \"${WANTED_STATUS}\" ]; do
        STATUS_CODE=\$(curl --write-out %{http_code} \
            --silent \
            --output /dev/null \
            ${URL})
        echo -n .
        sleep 1
      done"
}


function AppendAndJoin {
  local SUFFIX=${1}
  local SEPARATOR=${2}
  shift 2
  local ARRAY="$*"
  local o="$( printf "${SEPARATOR}%s${SUFFIX}" ${ARRAY} )"
  local o="${o:${#SEPARATOR}}" # remove leading separator
  echo "${o}"
}



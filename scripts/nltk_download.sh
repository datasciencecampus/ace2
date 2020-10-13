#!/usr/bin/env bash
#
# Download nltk data from HDFS
#
# To use in project settings set the following environment variable:
# NLTK_DATA=/home/cdsw/nltk_data
SCRIPT_NAME=nltk_data_download

NLTK_DIR=/home/cdsw/nltk_data
NLTK_TEMP_DIR=/home/cdsw/nltk_temp
NLTK_HDFS_DIR=/dap/landing_zone/natural_language_toolkit/nltk_corpora/3_4/v1/

function log {
    timestamp=`date "+%Y-%m-%d %H:%M:%S"`
    echo "[$SCRIPT_NAME] ${timestamp}: $1"
}

function nltk_download {
    
    log "downloading $1 $2"
    FILE="$NLTK_TEMP_DIR/${2}.zip"
    
    if [[ -e "${NLTK_DIR}/${1}/${2}" ]]; then
        log "$2 already in place"
        return 0
    fi
    
    if [[ -f "$FILE" ]]; then
        log "Already downloaded $2"
    else
        
        hdfs dfs -copyToLocal "$NLTK_HDFS_DIR${2}.zip" $NLTK_TEMP_DIR
        
        if [[ $? -ne 0 ]]; then
            log "nltk_download of ${2} failed."
            exit 1
        fi
    fi
    
    unzip "${NLTK_TEMP_DIR}/${2}.zip" -d $NLTK_TEMP_DIR
    cp -r "${NLTK_TEMP_DIR}/${2}" "${NLTK_DIR}/${1}"
}

FOLDERS=(
    chunkers
    grammars
    misc
    sentiment
    taggers
    corpora
    "help"
    models
    stemmers
    tokenizers    
)

mkdir -p $NLTK_TEMP_DIR

for folder in ${FOLDERS[@]}; do
    mkdir -p $NLTK_DIR/$folder
done

log "Downloading corpora"

nltk_download corpora stopwords
nltk_download corpora wordnet
nltk_download corpora wordnet_ic
nltk_download corpora sentiwordnet

log "Downloading tokenizers"

nltk_download tokenizers punkt

rm -rf $NLTK_TEMP_DIR



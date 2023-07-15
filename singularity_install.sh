#!/bin/bash

# fonts color
Green="\033[32m"
Red="\033[31m"
Yellow="\033[33m"
GreenBG="\033[42;37m"
RedBG="\033[41;37m"
Font="\033[0m"

WORK_PATH=$(dirname $(readlink -f $0))

echo -e "${Green}This a bash script for installing Singularity and DRLinFluids.${Font}"
echo -e "${Green}For more details, please refer to https://github.com/venturi123/DRLinFluids. ${Font}"

if [[ $(id -u) -ne 0 ]]; then
  echo -e "${Red}This script must be run as root. Exiting...${Font}"
  exit 1
fi

error_info() {
    echo -e "${Red}Unsupported system for quick installation.${Font}"
    echo -e "${Red}please refer to https://docs.sylabs.io/guides/latest/user-guide/quick_start.html# for installation.${Font}"
    exit 1
}

get_singularity_latest_version() {
    local OWNER="sylabs"
    local REPO="singularity"

    RESPONSE=$(curl -s "https://api.github.com/repos/$OWNER/$REPO/tags")

    if [ $? -ne 0 ]; then
        echo "Failed to get tags information for $REPO"
        exit 1
    fi

    LATEST_VERSION=$(echo "$RESPONSE" | grep -o '"name": "[^"]*"' | head -n1 | grep -Po '(\d+)(\.(\d+))+')

    if [ -z "$LATEST_VERSION" ]; then
        echo "Failed to get latest version for $REPO"
        exit 1
    fi

    echo "Latest version of $REPO is v$LATEST_VERSION"
}

install_singularity() {
    if [[ -e /etc/os-release ]]; then
        source /etc/os-release
        echo "Operating system is $NAME $VERSION."
    else
        error_info
    fi
    
    if [[ $(uname -m) != "x86_64" ]]; then
        error_info
    fi
    
    get_singularity_latest_version
    
    if [[ "$ID" == "ubuntu" || "$ID" == "debian" ]]; then
        SUFFIX="deb"
        PKG_INSTALLER="dpkg -i"
        DOWNLOAD_LINK="https://github.com/sylabs/singularity/releases/download/v${LATEST_VERSION}/singularity-ce_${LATEST_VERSION}-${VERSION_CODENAME}_amd64.deb"
        
        # check singularity
        if [ -f "/bin/singularity" ];then
            echo -e "${Yellow}Detected that Singularity is already installed.${Font}"
        else
            sudo apt update
            sudo apt -y install runc wget curl cryptsetup-bin uidmap
            wget ${DOWNLOAD_LINK} -O ${WORK_PATH}/singularity-ce_${LATEST_VERSION}.${SUFFIX}
            sudo ${PKG_INSTALLER} ${WORK_PATH}/singularity-ce_${LATEST_VERSION}.${SUFFIX}
            rm -rf ${WORK_PATH}/singularity-ce_${LATEST_VERSION}.${SUFFIX}
        fi
    
    elif  [[ "$ID" == "rhel" ||"$ID" == "centos" || "$ID" == "almalinux" || "$ID" == "rocky" ]]; then
        SUFFIX="rpm"
        PKG_INSTALLER="yum localinstall"
        DOWNLOAD_LINK="https://github.com/sylabs/singularity/releases/download/v${LATEST_VERSION}/singularity-ce-${LATEST_VERSION}-1.${VERSION_CODENAME}.${$PLATFORM_ID: -3}.rpm"
        # check singularity
        if [ -f "/bin/singularity" ];then
            echo -e "${Yellow}Detected that Singularity is already installed.${Font}"
        else
            sudo yum update
            sudo yum -y install runc wget curl cryptsetup-bin uidmap
            wget ${DOWNLOAD_LINK} -O ${WORK_PATH}/singularity-ce_${LATEST_VERSION}.${SUFFIX}
            sudo ${PKG_INSTALLER} ${WORK_PATH}/singularity-ce_${LATEST_VERSION}.${SUFFIX}
            rm -rf ${WORK_PATH}/singularity-ce_${LATEST_VERSION}.${SUFFIX}
        fi
        
    else
        error_info
    fi
}

install_singularity
rm -rf ${WORK_PATH}/singularity_install.sh

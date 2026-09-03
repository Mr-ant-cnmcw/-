#!/bin/bash

#quit when there is error
set -e

# 日志函数
log() { echo -e "[$(date '+%F %T')] [INFO] $1"; }
# 失败时自动打印错误日志
trap 'echo -e "[$(date '+%F %T')] [ERROR] 脚本执行失败"; exit 1' ERR

log "========== 开始配置环境 =========="

#environment setting bash
log "第 1 步: 更新软件源"
sudo apt update
log "第 1 步完成 ✓"

#basic tool
log "第 2 步: 安装基础工具 (git docker.io curl vim openssh-server fastfetch)"
sudo apt install -y git docker.io curl vim openssh-server fastfetch
log "第 2 步完成 ✓"

#python + pip
log "第 3 步: 安装 Python3 + pip"
sudo apt install -y python3 python3-pip
log "第 3 步完成 ✓"

#set locale
log "第 4 步: 配置中文 locale"
sudo apt install -y language-pack-zh-hans
sudo locale-gen zh_CN.UTF-8
sudo update-locale LANG=zh_CN.UTF-8
log "第 4 步完成 ✓"

#set timedate
log "第 5 步: 设置时区 Asia/Shanghai"
sudo timedatectl set-timezone Asia/Shanghai
log "第 5 步完成 ✓"

#update
log "第 6 步: 再次刷新软件源"
sudo apt update

log "========== 全部配置完成 ヽ(✿ﾟ▽ﾟ)ノ =========="

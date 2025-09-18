CREATE DATABASE IF NOT EXISTS `document_agent` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'agentusercenter'@'127.0.0.1' IDENTIFIED BY 'wdfamzwdfamz';
-- 确保密码被设置/重置（兼容已存在用户）
ALTER USER 'agentusercenter'@'127.0.0.1' IDENTIFIED BY 'wdfamzwdfamz';

GRANT ALL PRIVILEGES ON `document_agent`.* TO 'agentusercenter'@'127.0.0.1';
FLUSH PRIVILEGES;

use document_agent;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    username VARCHAR(50) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT uq_user_email UNIQUE (email),
    CONSTRAINT uq_user_username UNIQUE (username)
);

CREATE TABLE templates (
    id INT AUTO_INCREMENT PRIMARY KEY,  -- 自增主键
    name VARCHAR(255) NOT NULL,         -- 模板名称
    owner INT NOT NULL,                 -- 所有者（用户ID）
    content TEXT NOT NULL               -- 模板路径
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO templates (id, name, owner, content) VALUES (null, '中标结果公示', 0, 'static/formwork/中标结果公示.json');
INSERT INTO templates (id, name, owner, content) VALUES (null, '中标通知书', 0, 'static/formwork/中标通知书.json');
INSERT INTO templates (id, name, owner, content) VALUES (null, '招标公告', 0, 'static/formwork/招标公告.json');
INSERT INTO templates (id, name, owner, content) VALUES (null, '招标文件', 0, 'static/formwork/招标文件.json');
INSERT INTO templates (id, name, owner, content) VALUES (null, '资格预审文件', 0, 'static/formwork/资格预审文件.json');

CREATE TABLE jobs (
    job_id CHAR(64) PRIMARY KEY,         -- 任务ID，存放哈希值（如 SHA256，可用 CHAR(64) 或 VARCHAR）
    project_name VARCHAR(255) NOT NULL,  -- 项目名称
    user_id INT NOT NULL,                -- 用户ID
    template_id INT,                     -- 模板ID（关联 templates 表）
    knowledge_id VARCHAR(512),                    -- 知识库ID
    content TEXT,                        -- 任务内容
    file VARCHAR(512),                   -- 文件路径或文件名
    document TEXT,               -- 文档路径或文档名
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP  -- 创建时间
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE knowledge_bases (
    id INT AUTO_INCREMENT PRIMARY KEY,      -- 自增主键
    knowledge_bases_id CHAR(64) NOT NULL,   -- 知识库ID（可用哈希或UUID）
    owner INT NOT NULL,                     -- 所有者（用户ID）
    name VARCHAR(255) NOT NULL              -- 知识库名称
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE reviews (
    reviewsid CHAR(64) PRIMARY KEY,     -- 审核ID，哈希值，不自增
    job_id CHAR(64) NOT NULL,           -- 关联的任务ID
    type VARCHAR(50) NOT NULL,          -- 审核类型，比如 "content"、"document" 等
    original TEXT NOT NULL,             -- 原始内容
    modified TEXT,                      -- 修改后的内容
    reason TEXT,                        -- 修改/审核原因
    CONSTRAINT fk_reviews_jobs
        FOREIGN KEY (job_id) REFERENCES jobs(job_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE kb_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    knowledge_id INT NOT NULL,          -- 注意这里改名以示与 id 区分
    name VARCHAR(255) NOT NULL,
    hash VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    CONSTRAINT fk_kbfiles_kb
        FOREIGN KEY (knowledge_id) REFERENCES knowledge_bases(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE `user_document_numbers` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `user_id` INT NOT NULL,
    `prefix` VARCHAR(255) NOT NULL COMMENT '文号前缀 (公司简称+时间)',
    `number_part` VARCHAR(50) NOT NULL COMMENT '文号数字部分 (X号)',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户文号表';


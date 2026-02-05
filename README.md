# Oh my KEGG MCP

KEGG (Kyoto Encyclopedia of Genes and Genomes) 데이터베이스에 접근하기 위한 Model Context Protocol (MCP) 서버입니다.(unofficial)

## 주요 기능

- **30개의 KEGG 도구**: 경로, 유전자, 화합물, 반응, 효소, 질병, 약물 등 다양한 생물학적 데이터 검색 및 분석
- **Streamable HTTP Transport**: 웹 환경에서 사용 가능한 HTTP 기반 통신
- **LangChain 통합**: LangChain Agent와 완벽하게 통합되어 LLM이 KEGG 데이터를 활용할 수 있음
- **Ollama 지원**: 로컬 LLM 모델과 함께 사용 가능

## 설치

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env.example` 파일을 참고하여 `.env` 파일을 생성합니다:

```bash
cp .env.example .env
```

`.env` 파일을 편집하여 필요한 값을 설정합니다:

```bash
# OpenAI API 키 (OpenAI 모델 사용 시 필요)
# Ollama 사용 시에는 설정하지 않아도 됩니다
OPENAI_API_KEY=your_openai_api_key

# MCP 서버 URL (클라이언트에서 사용)
# 서버가 다른 호스트/포트에서 실행되는 경우 변경
KEGG_MCP_SERVER_URL=http://localhost:3000/mcp

# MCP 서버 설정 (서버 실행 시 사용, 선택사항)
# 기본값이 설정되어 있어 변경하지 않아도 됩니다
MCP_HOST=localhost
MCP_PORT=3000
MCP_PATH=/mcp

# Ollama 설정 (선택사항)
# Ollama가 다른 호스트/포트에서 실행되는 경우 변경
# OLLAMA_HOST=http://localhost:11434
```

**참고**: 
- **Ollama 사용 시**: `OPENAI_API_KEY`는 설정하지 않아도 됩니다. 현재 코드는 Ollama를 기본 모델로 사용합니다.
- **OpenAI 사용 시**: `OPENAI_API_KEY`에 실제 API 키를 설정하고, `langchain_integration.py`에서 `ChatOpenAI`를 사용하도록 변경해야 합니다.
- **서버 설정**: `MCP_HOST`, `MCP_PORT`, `MCP_PATH`는 서버 실행 시 사용되며, 기본값이 설정되어 있어 변경하지 않아도 됩니다.
- **Ollama 호스트**: Ollama가 기본 포트(11434)가 아닌 다른 포트에서 실행되는 경우 `OLLAMA_HOST`를 설정할 수 있습니다.

## 사용 방법

### 1. 서버 실행

```bash
# 기본 설정으로 실행 (localhost:3000/mcp)
python kegg_mcp_server.py

# 또는 환경 변수로 설정
MCP_HOST=localhost MCP_PORT=3000 MCP_PATH=/mcp python kegg_mcp_server.py
```

서버가 실행되면 다음과 같은 메시지가 표시됩니다:
```
Starting KEGG MCP Server on http://localhost:3000/mcp
Transport: Streamable HTTP
```

### 2. 클라이언트 실행

```bash
# 기본 실행 (Streamable HTTP 사용)
python langchain_integration.py

# Ollama 연결 테스트만 실행
python langchain_integration.py --test

# stdio 사용 (선택사항, 별도 TypeScript 서버 필요)
# langchain_integration.py에서 use_http=False로 변경
# 참고: 현재는 Streamable HTTP 방식만 제공됩니다
```

## 사용 가능한 도구

### Database Information & Statistics (2개)
- `get_database_info` - 데이터베이스 정보 및 통계 조회
- `list_organisms` - 모든 KEGG 생물체 목록 조회

### Pathway Analysis (3개)
- `search_pathways` - 경로 검색
- `get_pathway_info` - 경로 상세 정보
- `get_pathway_genes` - 경로의 유전자 목록

### Gene Analysis (2개)
- `search_genes` - 유전자 검색
- `get_gene_info` - 유전자 상세 정보 (서열 포함 옵션)

### Compound Analysis (2개)
- `search_compounds` - 화합물 검색
- `get_compound_info` - 화합물 상세 정보

### Reaction & Enzyme Analysis (4개)
- `search_reactions` - 반응 검색
- `get_reaction_info` - 반응 상세 정보
- `search_enzymes` - 효소 검색
- `get_enzyme_info` - 효소 상세 정보

### Disease & Drug Analysis (5개)
- `search_diseases` - 질병 검색
- `get_disease_info` - 질병 상세 정보
- `search_drugs` - 약물 검색
- `get_drug_info` - 약물 상세 정보
- `get_drug_interactions` - 약물 상호작용 조회

### Module & Orthology Analysis (4개)
- `search_modules` - 모듈 검색
- `get_module_info` - 모듈 상세 정보
- `search_ko_entries` - KO 엔트리 검색
- `get_ko_info` - KO 상세 정보

### Glycan Analysis (2개)
- `search_glycans` - 글리칸 검색
- `get_glycan_info` - 글리칸 상세 정보

### BRITE Hierarchy Analysis (2개)
- `search_brite` - BRITE 계층 검색
- `get_brite_info` - BRITE 상세 정보

### Advanced Analysis Tools (5개)
- `get_pathway_compounds` - 경로의 화합물 목록
- `get_pathway_reactions` - 경로의 반응 목록
- `get_compound_reactions` - 화합물의 반응 목록
- `get_gene_orthologs` - 유전자 직계동족 찾기
- `batch_entry_lookup` - 배치 엔트리 조회

### Cross-References & Integration (2개)
- `convert_identifiers` - 데이터베이스 간 식별자 변환
- `find_related_entries` - 관련 엔트리 찾기

## 예제

### 기본 사용 예제

```python
import asyncio
from langchain_integration import create_kegg_client, load_and_display_tools, create_default_model, run_agent_query
from langchain.agents import create_agent

async def main():
    # 클라이언트 생성 (Streamable HTTP)
    client = create_kegg_client()
    
    try:
        # 도구 로드
        tools = await load_and_display_tools(client)
        
        # 모델 및 Agent 생성
        model = create_default_model()
        agent = create_agent(model=model, tools=tools)
        
        # 쿼리 실행
        await run_agent_query(
            agent,
            "인간의 당분해(glycolysis) 경로를 찾아주세요.",
            "당분해 경로 검색"
        )
    finally:
        if hasattr(client, 'close'):
            try:
                await client.close()
            except:
                pass

if __name__ == "__main__":
    asyncio.run(main())
```

## 프로젝트 구조

```
kegg-mcp-test/
├── kegg_mcp_server.py          # Python MCP 서버 (Streamable HTTP)
├── langchain_integration.py    # LangChain 통합 클라이언트
├── requirements.txt             # Python 패키지 의존성
├── .env.example                # 환경 변수 예제 파일
├── .gitignore                  # Git 무시 파일 목록
├── LICENSE                     # MIT 라이선스
└── README.md                    # 프로젝트 문서 (이 파일)
```

## 통신 방식

### Streamable HTTP (권장)
- **장점**: 웹 환경 지원, 확장성 좋음, 표준 HTTP 프로토콜
- **사용**: Python 서버 (`kegg_mcp_server.py`)
- **포트**: 기본 3000 (환경 변수로 변경 가능)

### stdio (로컬 전용, 선택사항)
- **장점**: 최고의 보안, 낮은 오버헤드, 간단한 설정
- **사용**: `langchain_integration.py`에서 `use_http=False`로 설정 시 사용 가능
- **제한**: 로컬 실행만 가능, 별도 TypeScript 서버 필요

## 문제 해결

### 서버가 시작되지 않는 경우
1. 포트가 이미 사용 중인지 확인: `lsof -i :3000`
2. 필요한 패키지가 설치되었는지 확인: `pip install -r requirements.txt`
3. Python 버전 확인: Python 3.11 이상 필요

### 클라이언트 연결 실패
1. 서버가 실행 중인지 확인
2. 서버 URL이 올바른지 확인: `KEGG_MCP_SERVER_URL` 환경 변수
3. 방화벽 설정 확인

### Ollama 연결 실패
1. Ollama가 실행 중인지 확인: `ollama serve`
2. 모델이 설치되었는지 확인: `ollama list`
3. 모델 다운로드: `ollama pull gemma3:4b` (또는 사용할 모델)

## 라이선스

MIT License

## 참고 자료

- [KEGG REST API 문서](https://www.kegg.jp/kegg/rest/keggapi.html)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [LangChain 문서](https://python.langchain.com)
- [FastMCP 문서](https://github.com/jlowin/fastmcp)

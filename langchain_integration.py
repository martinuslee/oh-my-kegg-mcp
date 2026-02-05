"""
LangChain Python과 KEGG MCP 서버 연동 예제

이 스크립트는 KEGG MCP 서버를 LangChain agent에 통합하는 방법을 보여줍니다.
"""
import os
import asyncio
from pathlib import Path
from typing import Any, Callable, Optional

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

from dotenv import load_dotenv
load_dotenv()

# 설정 상수
KEGG_SERVER_PATH = Path(__file__).parent / "KEGG-MCP-Server" / "build" / "index.js"
DEFAULT_MODEL = "granite4:1b"
DEFAULT_TEMPERATURE = 0
DEFAULT_BASE_URL = "http://localhost:11434"  # Ollama 기본 URL
KEGG_MCP_SERVER_URL = os.getenv("KEGG_MCP_SERVER_URL", "http://localhost:3000/mcp")  # MCP 서버 URL (Streamable HTTP)
MAX_TOOLS_TO_DISPLAY = 10


def create_kegg_client(
    tool_interceptors: Optional[list[Callable]] = None,
    use_http: bool = True
) -> MultiServerMCPClient:
    """
    KEGG MCP 서버 클라이언트를 생성합니다.
    
    Args:
        tool_interceptors: 도구 호출 인터셉터 리스트 (선택사항)
        use_http: True면 Streamable HTTP 사용, False면 stdio 사용 (기본값: True)
    
    Returns:
        MultiServerMCPClient 인스턴스
    """
    if use_http:
        # Streamable HTTP 방식 사용 (Python 서버)
        config = {
            "kegg": {
                "transport": "streamable-http",  # 또는 "http"
                "url": KEGG_MCP_SERVER_URL,
            }
        }
    else:
        # stdio 방식 사용 (TypeScript 서버)
        config = {
            "kegg": {
                "transport": "stdio",
                "command": "node",
                "args": [str(KEGG_SERVER_PATH.absolute())],
            }
        }
    
    kwargs = {}
    if tool_interceptors:
        kwargs["tool_interceptors"] = tool_interceptors
    
    return MultiServerMCPClient(config, **kwargs)


async def load_and_display_tools(client: MultiServerMCPClient) -> list[Any]:
    """
    MCP 서버에서 도구를 로드하고 정보를 출력합니다.
    
    Args:
        client: MultiServerMCPClient 인스턴스
    
    Returns:
        로드된 도구 리스트
    """
    print("KEGG MCP 서버에서 도구를 로드하는 중...")
    tools = await client.get_tools()
    print(f"로드된 도구 수: {len(tools)}")
    
    print("\n사용 가능한 도구:")
    for tool in tools[:MAX_TOOLS_TO_DISPLAY]:
        description = tool.description[:80] if tool.description else "설명 없음"
        print(f"  - {tool.name}: {description}...")
    
    if len(tools) > MAX_TOOLS_TO_DISPLAY:
        print(f"  ... 및 {len(tools) - MAX_TOOLS_TO_DISPLAY}개 더")
    
    return tools


def create_default_model() -> ChatOllama:
    """
    기본 Ollama 모델을 생성합니다.
    
    Returns:
        ChatOllama 인스턴스
    """
    return ChatOllama(
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        base_url=DEFAULT_BASE_URL
    )


async def run_agent_query(
    agent: Any,
    query: str,
    example_name: str
) -> str:
    """
    Agent를 사용하여 쿼리를 실행하고 결과를 반환합니다.
    
    Args:
        agent: LangChain Agent 인스턴스 (create_agent로 생성)
        query: 실행할 쿼리
        example_name: 예제 이름 (출력용)
    
    Returns:
        Agent 응답 내용
    """
    print("\n" + "=" * 60)
    print(f"예제: {example_name}")
    print("=" * 60)
    
    response = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": query
        }]
    })
    
    content = response["messages"][-1].content
    print(content)
    return content


async def test_ollama_connection():
    """
    Ollama 연결 및 응답 테스트 (MCP 서버 없이)
    """
    print("=" * 60)
    print("Ollama 연결 테스트")
    print("=" * 60)
    
    try:
        model = create_default_model()
        print(f"모델: {DEFAULT_MODEL}")
        print(f"Ollama URL: {DEFAULT_BASE_URL}")
        
        # 간단한 테스트 쿼리
        print("\n테스트 쿼리 전송 중...")
        response = await model.ainvoke("안녕하세요! 간단히 자기소개 해주세요. 한 문장으로 답변해주세요.")
        
        print("\n응답 받음:")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
        print("\n✅ Ollama 연결 성공!")
        return True
        
    except Exception as e:
        print(f"\n❌ Ollama 연결 실패: {e}")
        print("\n확인 사항:")
        print("1. Ollama가 설치되어 있고 실행 중인지 확인 (ollama serve)")
        print("2. 모델이 설치되어 있는지 확인 (ollama list)")
        print(f"3. 모델 '{DEFAULT_MODEL}'이 설치되어 있는지 확인 (ollama pull {DEFAULT_MODEL})")
        print(f"4. Ollama 서버가 {DEFAULT_BASE_URL}에서 실행 중인지 확인")
        return False


async def main():
    """
    KEGG MCP 서버를 LangChain agent에 통합하는 메인 함수
    """
    client = create_kegg_client()
    
    try:
        tools = await load_and_display_tools(client)
        model = create_default_model()
        agent = create_agent(model=model, tools=tools)
        
        # 예제 쿼리 실행
        await run_agent_query(
            agent,
            "인간의 당분해(glycolysis) 경로를 찾아주세요. 경로 ID와 이름을 알려주세요.",
            "인간의 당분해 경로 검색"
        )
        
        await run_agent_query(
            agent,
            "인간(hsa)의 인슐린(insulin) 유전자를 검색하고 상세 정보를 알려주세요.",
            "인슐린 유전자 정보 조회"
        )
        
        await run_agent_query(
            agent,
            "글루코스(glucose) 화합물을 검색하고 KEGG ID와 분자식을 알려주세요.",
            "글루코스 화합물 검색"
        )
    finally:
        # MultiServerMCPClient는 자동으로 정리됨
        if hasattr(client, 'close'):
            try:
                await client.close()
            except Exception:
                pass

async def example_with_stateful_session():
    """
    Stateful 세션을 사용하는 예제 (여러 도구 호출 간 상태 유지)
    """
    from langchain_mcp_adapters.tools import load_mcp_tools
    
    client = create_kegg_client()
    
    try:
        async with client.session("kegg") as session:
            tools = await load_mcp_tools(session)
            model = create_default_model()
            agent = create_agent(model=model, tools=tools)
            
            await run_agent_query(
                agent,
                "인간의 당분해 경로(hsa00010)에 포함된 모든 유전자를 찾아주세요.",
                "당분해 경로 유전자 조회 (Stateful 세션)"
            )
    finally:
        # MultiServerMCPClient는 자동으로 정리됨
        if hasattr(client, 'close'):
            try:
                await client.close()
            except Exception:
                pass


async def example_with_interceptors():
    """
    Tool interceptors를 사용하여 요청/응답을 수정하는 예제
    """
    async def logging_interceptor(request: MCPToolCallRequest, handler):
        """도구 호출을 로깅하는 인터셉터"""
        print(f"[INTERCEPTOR] 도구 호출: {request.name}")
        print(f"[INTERCEPTOR] 인자: {request.args}")
        
        result = await handler(request)
        
        print(f"[INTERCEPTOR] 결과: {str(result)[:100]}...")
        return result
    
    client = create_kegg_client(tool_interceptors=[logging_interceptor])
    
    try:
        tools = await client.get_tools()
        model = create_default_model()
        agent = create_agent(model=model, tools=tools)
        
        await run_agent_query(
            agent,
            "인간의 당분해 경로 정보를 알려주세요.",
            "당분해 경로 조회 (인터셉터 사용)"
        )
    finally:
        # MultiServerMCPClient는 자동으로 정리됨
        if hasattr(client, 'close'):
            try:
                await client.close()
            except Exception:
                pass


def print_error_help():
    """에러 발생 시 도움말을 출력합니다."""
    print("\n다음 사항을 확인하세요:")
    print("1. Node.js가 설치되어 있고 PATH에 있는지 확인")
    print("2. KEGG 서버가 빌드되었는지 확인 (npm run build)")
    print("3. 필요한 Python 패키지가 설치되었는지 확인:")
    print("   pip install langchain-mcp-adapters langchain-ollama")
    print("4. Ollama가 설치되어 있고 실행 중인지 확인 (ollama serve)")
    print(f"5. 모델 '{DEFAULT_MODEL}'이 설치되어 있는지 확인 (ollama pull {DEFAULT_MODEL})")
    print(f"6. MCP 서버가 HTTP 모드로 실행 중인지 확인 (서버 URL: {KEGG_MCP_SERVER_URL})")
    print("   - 환경 변수 KEGG_MCP_SERVER_URL로 서버 URL 설정 가능")
    print("   - Python 서버 실행: python kegg_mcp_server.py")
    print("   - 또는 TypeScript 서버 사용: use_http=False로 설정")


if __name__ == "__main__":
    import sys
    
    # 명령줄 인자로 테스트 모드 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Ollama 연결 테스트만 실행
        print("Ollama 연결 테스트 모드")
        print("=" * 60)
        try:
            success = asyncio.run(test_ollama_connection())
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            print("\n프로그램이 중단되었습니다.")
            sys.exit(1)
        except Exception as e:
            print(f"\n오류 발생: {e}")
            sys.exit(1)
    else:
        # 기본 실행: 먼저 Ollama 테스트 후 메인 실행
        print("KEGG MCP 서버와 LangChain 통합 예제 (Ollama)")
        print("=" * 60)
        
        # 먼저 Ollama 연결 테스트
        print("\n[1단계] Ollama 연결 테스트")
        try:
            test_success = asyncio.run(test_ollama_connection())
            if not test_success:
                print("\n⚠️  Ollama 연결 실패. 메인 프로그램을 계속 실행합니다...")
        except Exception as e:
            print(f"\n⚠️  Ollama 테스트 중 오류: {e}")
        
        # 메인 프로그램 실행
        print("\n[2단계] KEGG MCP 서버 통합 실행")
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n프로그램이 중단되었습니다.")
        except Exception as e:
            print(f"\n오류 발생: {e}")
            print_error_help()

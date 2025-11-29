#!/usr/bin/env python3
"""
快速测试: 验证Programmer operator和response_standardizer修复
"""
import asyncio
import sys
sys.path.insert(0, '/home/yijia/.claude/11/AFlow')
sys.path.insert(0, '/home/yijia/.claude/11/integrated_aflow_roll/src')

from response_standardizer import ResponseStandardizer

def test_response_standardizer():
    """测试response_standardizer对Programmer返回值的处理"""
    print("=" * 60)
    print("测试1: ResponseStandardizer对Programmer返回值的处理")
    print("=" * 60)

    # 模拟Programmer的返回值
    programmer_response = {
        "code": "def solve():\n    return 21 * 2 + 0\n",
        "output": "42"
    }

    # 标准化
    standardized = ResponseStandardizer.standardize(programmer_response, "Programmer")

    print(f"\n原始Programmer返回值:")
    print(f"  code: {repr(programmer_response['code'][:50])}...")
    print(f"  output: {repr(programmer_response['output'])}")

    print(f"\n标准化后:")
    print(f"  content (应该是执行结果): {repr(standardized['content'])}")
    print(f"  metadata['code'] (应该是源码): {repr(standardized['metadata'].get('code', 'N/A')[:30])}...")

    # 验证
    if standardized['content'] == "42":
        print("\n✅ 测试通过: content正确返回了执行结果 '42'")
        return True
    else:
        print(f"\n❌ 测试失败: content应该是 '42', 但实际是 {repr(standardized['content'])}")
        return False


async def test_programmer_execution():
    """测试真实的Programmer执行"""
    print("\n" + "=" * 60)
    print("测试2: 真实Programmer执行")
    print("=" * 60)

    try:
        from scripts.operators import run_code

        # 测试简单数学计算
        test_code = '''
def solve():
    result = 21 * 2 + 0
    return result
'''

        status, output = run_code(test_code)
        print(f"\n执行代码:")
        print(f"  {test_code.strip()}")
        print(f"\n执行结果:")
        print(f"  status: {status}")
        print(f"  output: {repr(output)}")

        if status == "Success" and output == "42":
            print("\n✅ 测试通过: run_code正确执行并返回结果 '42'")
            return True
        else:
            print(f"\n❌ 测试失败: 期望 ('Success', '42'), 实际 ({status}, {repr(output)})")
            return False

    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_workflow_simulation():
    """模拟完整的工作流执行流程"""
    print("\n" + "=" * 60)
    print("测试3: 模拟完整工作流 (Programmer -> 取output作为答案)")
    print("=" * 60)

    try:
        from scripts.operators import run_code

        # 模拟工作流中使用Programmer
        problem = "Calculate 21 * 2 + 0"

        # Step 1: 生成代码 (模拟LLM生成)
        generated_code = '''
def solve():
    # Calculate 21 * 2 + 0
    result = 21 * 2 + 0
    return result
'''

        # Step 2: 执行代码 (run_code)
        status, output = run_code(generated_code)

        # Step 3: 构造Programmer返回值
        programmer_result = {
            "code": generated_code,
            "output": output
        }

        # Step 4: 标准化 (可选，取决于是否使用标准化)
        standardized = ResponseStandardizer.standardize(programmer_result, "Programmer")

        # Step 5: 正确取值 - 使用output而非code
        final_answer_correct = programmer_result['output']  # ✅ 正确
        final_answer_wrong = programmer_result['code']       # ❌ 错误

        print(f"\n问题: {problem}")
        print(f"\n执行状态: {status}")
        print(f"\n正确答案 (result['output']): {repr(final_answer_correct)}")
        print(f"错误答案 (result['code']): {repr(final_answer_wrong[:30])}...")
        print(f"\n标准化后content: {repr(standardized['content'])}")

        if final_answer_correct == "42" and standardized['content'] == "42":
            print("\n✅ 测试通过: 工作流能正确获取执行结果")
            return True
        else:
            print("\n❌ 测试失败")
            return False

    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "#" * 60)
    print("# Programmer修复验证测试")
    print("#" * 60)

    results = []

    # 测试1: ResponseStandardizer
    results.append(("ResponseStandardizer", test_response_standardizer()))

    # 测试2 & 3: 需要异步
    loop = asyncio.get_event_loop()
    results.append(("run_code执行", loop.run_until_complete(test_programmer_execution())))
    results.append(("完整工作流模拟", loop.run_until_complete(test_full_workflow_simulation())))

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 所有测试通过! Programmer修复验证成功!")
    else:
        print("\n⚠️  部分测试失败，请检查修复")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

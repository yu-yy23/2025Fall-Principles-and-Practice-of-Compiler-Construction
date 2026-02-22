# Parser Stage

此分支专用于 parser-stage。设立这个分支的目的在于排除 parser-stage 中用不到的干扰。

请用 frontend/ 中的文件覆盖你已有的编译器（记得先 git checkout parser-stage），并补全 TODO 部分。

可以使用 [test-parser-stage.sh](./test-parser-stage.sh) 来测试自己写的 parser 在 step 1-6 的测例下有没有不正常地报错退出。

测试生成的 AST 的正确性可以采取 ~~肉眼瞪~~ 正确完成 stage2 后将 frontend/parser 替换为 parser-stage 中对应的文件夹并运行 minidecaf-tests 中的测试来实现。注意这也是最后 CI 中用于评判实现正确性的办法。

## 依赖

- **Python >= 3.9**
- requirements.txt 里的 python 库 argparse。

## 运行

```
python3 main.py --input <testcase.c> --parse
```

各参数意义如下：

| 参数 | 含义 |
| --- | --- |
| `input` | 输入的 Minidecaf 代码位置 |
| `parse` | 输出抽象语法树 |


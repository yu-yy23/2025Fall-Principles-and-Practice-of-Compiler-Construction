from __future__ import annotations

from typing import Callable, Protocol

from frontend.ast.tree import *
from frontend.lexer import Lexer, LexToken, lexer
from utils.error import DecafSyntaxError

co_T = TypeVar("co_T", covariant=True)


class Rule(Protocol[co_T]):
    first: frozenset[str]

    def __call__(self, parser: Parser) -> co_T:
        ...


def first(*first: str):
    def decorator(f: Callable[[Parser], T]) -> Rule[T]:
        f.first = frozenset(first)
        return f

    return decorator


@first("LParen", "Identifier", "Integer")
def p_primary_expression(self: Parser) -> Expression:
    """
    primary : Integer
        | Identifier
        | '(' expression ')'
    """
    lookahead = self.lookahead
    if self.next in ("Identifier", "Integer"):
        return lookahead()
    elif self.next == "LParen":
        lookahead()
        expr = p_expression(self)
        lookahead("RParen")
        return expr
    raise DecafSyntaxError(self.next_token)


@first("Minus", "Not", "BitNot", *p_primary_expression.first)
def p_unary(self: Parser) -> Expression:
    """
    unary : Minus unary
        | BitNot unary
        | Not unary
        | primary
    """
    lookahead = self.lookahead
    if self.next in p_primary_expression.first:
        return p_primary_expression(self)
    elif self.next in ("Minus", "Not", "BitNot"):
        op = UnaryOp.backward_search(self.lookahead())
        oprand = p_unary(self)
        return Unary(op, oprand)
    raise DecafSyntaxError(self.next_token)


@first(*p_unary.first)
def p_multiplicative(self: Parser) -> Expression:
    """
    multiplicative : multiplicative '*' unary
        | multiplicative '/' unary
        | multiplicative '%' unary
        | unary
    """

    """
    equivalent EBNF:
    multiplicative: unary { '*' unary | '/' unary | '%' unary }
    """

    lookahead = self.lookahead
    node = p_unary(self)
    while self.next in ("Mul", "Div", "Mod"):
        op = BinaryOp.backward_search(lookahead())
        rhs = p_unary(self)
        node = Binary(op, node, rhs)
    return node


@first(*p_multiplicative.first)
def p_additive(self: Parser) -> Expression:
    """
    additive : additive '+' multiplicative
        | additive '-' multiplicative
        | multiplicative
    """

    """
    equivalent EBNF:
    additive: multiplicative { '+' multiplicative | '-' multiplicative }
    """
    lookahead = self.lookahead
    node = p_multiplicative(self)
    while self.next in ("Plus", "Minus"):
        op = BinaryOp.backward_search(lookahead())
        rhs = p_multiplicative(self)
        node = Binary(op, node, rhs)
    return node


@first(*p_additive.first)
def p_relational(self: Parser) -> Expression:
    """
    relational : relational '<' additive
        | relational '>' additive
        | relational '<=' additive
        | relational '>=' additive
        | additive
    """

    """ TODO
    1. Refer to the implementation of 'p_equality'.
    """
    
    """
    equivalent EBNF:
    relational: additive { '<' additive | '>' additive | '<=' additive | '>=' additive }
    """

    lookahead = self.lookahead
    node = p_additive(self)
    while self.next in ("Less", "Greater", "LessEqual", "GreaterEqual"):
        op = BinaryOp.backward_search(lookahead())
        rhs = p_additive(self)
        node = Binary(op, node, rhs)
    return node


@first(*p_relational.first)
def p_equality(self: Parser) -> Expression:
    """
    equality : equality '==' relational
        | equality '!=' relational
        | relational
    """

    """
    equivalent EBNF:
    equality: relational { '==' relational | '!=' relational }
    """
    lookahead = self.lookahead
    node = p_relational(self)
    while self.next in ("Equal", "NotEqual"):
        op = BinaryOp.backward_search(lookahead())
        rhs = p_relational(self)
        node = Binary(op, node, rhs)
    return node


@first(*p_equality.first)
def p_logical_and(self: Parser) -> Expression:
    """
    logical_and : logical_and '&&' equality
        | equality
    """

    """ TODO
    1. Refer to the implementation of 'p_logical_or'.
    """
    
    """
    equivalent EBNF:
    logical_and: equality { '&&' equality }
    """

    lookahead = self.lookahead
    node = p_equality(self)
    while self.next in ("And",):
        op = BinaryOp.backward_search(lookahead())
        rhs = p_equality(self)
        node = Binary(op, node, rhs)
    return node


@first(*p_logical_and.first)
def p_logical_or(self: Parser) -> Expression:
    """
    logical_or : logical_or '||' logical_and
        | logical_and
    """

    """
    equivalent EBNF:
    logical_or: logical_and { '||' logical_and }
    """
    lookahead = self.lookahead
    node = p_logical_and(self)
    while self.next in ("Or",):
        op = BinaryOp.backward_search(lookahead())
        rhs = p_logical_and(self)
        node = Binary(op, node, rhs)
    return node


@first(*p_logical_or.first)
def p_conditional(self: Parser) -> Expression:
    """
    conditional : logical_or '?' expression ':' conditional
        | logical_or
    """
    lookahead = self.lookahead
    cond = p_logical_or(self)
    if self.next == "Question":
        lookahead()
        then = p_expression(self)
        lookahead("Colon")
        otherwise = p_conditional(self)
        return ConditionExpression(cond, then, otherwise)
    else:
        return cond


@first(*p_conditional.first)
def p_assignment(self: Parser) -> Expression:
    """
    assignment : Identifier '=' expression
        | conditional
    """
    lookahead = self.lookahead
    node_type = self.next
    current_tok = self.next_token
    node = p_conditional(self)
    if self.next == "Assign":
        if node_type != "Identifier":
            raise DecafSyntaxError(current_tok)
        """ TODO
        1. Match token 'Assign'.
        2. Parse expression to get rhs.
        3. Build an `Assignment` node with node (as lhs) and rhs
        4. Return the node.
        """
        lookahead("Assign")
        rhs = p_expression(self)
        node = Assignment(node, rhs)
        return node
    else:
        return node


@first(*p_assignment.first)
def p_expression(self: Parser) -> Expression:
    """
    expression : assignment
    """

    """ TODO
    1. Parse assignment and return it.
    """
    
    node = p_assignment(self)
    return node


@first("Return", *p_expression.first, "Semi", "LBrace")  # TODO fill in first set
def p_statement(self: Parser) -> Statement:
    "statement : return | ( expression )? ';' | block"

    if self.next in p_expression.first:
        expr = p_expression(self)
        self.lookahead("Semi")
        return expr
    elif self.next == "Semi":
        self.lookahead()
        return NULL  # type: ignore
    elif self.next == "LBrace":
        return p_block(self)
    elif self.next == "Return":
        return p_return(self)

    raise DecafSyntaxError(self.next_token)

    """ TODO
    1. Call the corresponding parsing function and return its result if `self.next` in p_return.first/p_block.first
    2. Otherwise just raise error as below.
    """


@first("Int")
def p_declaration(self: Parser) -> Declaration:
    "declaration : type Identifier ('=' expression)?"
    lookahead = self.lookahead
    var_t = p_type(self)
    ident = lookahead("Identifier")
    decl = Declaration(var_t, ident)
    if self.next == "Assign":
        """TODO
        1. Match token 'Assign'.
        2. Parse expression to get the initial value.
        3. Set the child `init_expr` of `decl`.
        """
        lookahead("Assign")
        init_expr = p_expression(self)
        decl.init_expr = init_expr
    return decl


@first("LBrace")
def p_block(self: Parser) -> Block:
    "block : '{' (statement | declaration ';')* '}'"

    def p_block_item(self: Parser) -> Union[Statement, Declaration]:
        "TODO: Return either a `Statement` or a `Declaration` that represents the next block_item in block"
        if self.next in p_statement.first:
            # TODO: Complete the action if the next is a statement.
            return p_statement(self)
        elif self.next in p_declaration.first:
            # TODO: Complete the action if the next is a declaration.
            node = p_declaration(self)
            self.lookahead("Semi")
            return node
        else:
            raise DecafSyntaxError(self.next_token)

    block = Block()
    lookahead = self.lookahead
    lookahead("LBrace")
    while self.next != "RBrace":
        item = p_block_item(self)
        block.children.append(item)
    lookahead("RBrace")
    return block

    """ TODO
	1. Match token 'LBrace'
	2. use `p_block_item` to parse block item until `RBrace` is met
	3. Match token 'RBrace'
	"""


def p_return(self: Parser) -> Return:
    "return : 'return' expression ';'"

    """ TODO
    1. Match token 'Return'.
    2. Parse expression.
    3. Match token 'Semi'.
    4. Build a `Return` node and return it.
    """
    
    lookahead = self.lookahead
    lookahead("Return")
    expr = p_expression(self)
    lookahead("Semi")
    return Return(expr)


def p_type(self: Parser) -> TypeLiteral:
    "type : 'int'"

    """ TODO
    1. Match token 'Int'.
    2. Build a `TInt` node and return it.
    """
    
    lookahead = self.lookahead
    lookahead("Int")
    return TInt()


def p_program(self: Parser) -> Program:
    "program : type Identifier '(' ')' block"
    lookahead = self.lookahead
    ret_t = p_type(self)
    ident = lookahead("Identifier")
    lookahead("LParen")
    lookahead("RParen")
    body = p_block(self)
    tail = lookahead()
    if tail is not None:
        raise DecafSyntaxError(tail)
    return Program(Function(ret_t, ident, body))


class Parser:
    def __init__(self, _lexer: Optional[Lexer] = None) -> None:
        self.lexer = _lexer or lexer
        self.next_token: Optional[LexToken]
        self.error_stack = list[DecafSyntaxError]()

    def lookahead(self, type: Optional[str] = None) -> Any:
        tok = self.next_token
        if tok is None:
            return tok
        if tok.type == type or type is None:
            try:
                self.next_token = next(self.lexer)
            except StopIteration:
                self.next_token = None
            return tok and tok.value
        raise DecafSyntaxError(tok)

    @property
    def next(self):
        if self.next_token is None:
            raise StopIteration
        return self.next_token.type

    def parse(self, input: str, lexer: Optional[Lexer] = None):
        if lexer:
            self.lexer = lexer
        self.lexer.input(input)
        self.next_token = next(self.lexer)
        # try:
        return p_program(self)
        # except DecafSyntaxError as e:
        #     self.error_stack.append(e)


parser = Parser()

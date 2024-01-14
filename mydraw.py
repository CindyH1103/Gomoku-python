import pygame
import sys
import time

SIZE = 90  # 棋盘每个点之间的间隔
Line_Points = 9  # 棋盘每行/每列点数
Outer_Width = 60  # 棋盘外宽度
Border_Width = 12  # 边框宽度
Inside_Width = 12  # 边框跟实际的棋盘之间的间隔
Border_Length = SIZE * (Line_Points - 1) + Inside_Width * 2 + Border_Width  # 边框线的长度
Start_X = Start_Y = Outer_Width + int(Border_Width / 2) + Inside_Width  # 网格线起点（左上角）坐标
SCREEN_HEIGHT = SIZE * (Line_Points - 1) + Outer_Width * 2 + Border_Width + Inside_Width * 2  # 游戏屏幕的高
SCREEN_WIDTH = SCREEN_HEIGHT + 600  # 游戏屏幕的宽
Text_X = Start_X + Border_Length + 30  # 文字的坐标
Text_Y = Start_Y
Line_Space = 120  # 文字行间距

Stone_Radius = SIZE // 2 - 9  # 棋子半径
Stone_Radius2 = SIZE // 4  # 图例棋子半径
Checkerboard_Color = (0xE3, 0x92, 0x65)  # 棋盘颜色
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
RED_COLOR = (200, 30, 30)
BLUE_COLOR = (30, 30, 200)


# 画棋盘
def _draw(poses, winners, Line_Points):
    color = [(0, 0, 0), (255, 255, 255)]
    pygame.init()
    my_font = pygame.font.SysFont("arial", 60)
    my_font_small = pygame.font.SysFont("arial", 40)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    done = False
    clock = pygame.time.Clock()
    a = b = 0
    while not done:
        clock.tick(10)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        # 填充棋盘背景色
        screen.fill(Checkerboard_Color)
        # 画棋盘网格线外的边框
        pygame.draw.rect(screen, BLACK_COLOR, (Outer_Width, Outer_Width, Border_Length, Border_Length), Border_Width)
        # 画网格线
        for i in range(Line_Points):
            pygame.draw.line(screen, BLACK_COLOR,
                             (Start_Y, Start_Y + SIZE * i),
                             (Start_Y + SIZE * (Line_Points - 1), Start_Y + SIZE * i),
                             1)
        for j in range(Line_Points):
            pygame.draw.line(screen, BLACK_COLOR,
                             (Start_X + SIZE * j, Start_X),
                             (Start_X + SIZE * j, Start_X + SIZE * (Line_Points - 1)),
                             1)

        # 添加文字信息
        # 轮数
        text_round = my_font.render("Round " + str(a + 1), True, color[0])
        screen.blit(text_round, (Text_X, Text_Y))
        # 图例
        text_PCplayer = my_font_small.render("MCTS_Pure", True, color[0])
        text_Ourplayer = my_font_small.render("Our Player", True, color[0])
        pygame.draw.circle(screen, color[0], [Text_X + Stone_Radius2, Text_Y + Line_Space], Stone_Radius2)
        pygame.draw.circle(screen, color[1], [Text_X + Stone_Radius2, Text_Y + Line_Space * 1.5], Stone_Radius2)
        screen.blit(text_Ourplayer, (Text_X + Stone_Radius2 * 3, Text_Y + Line_Space - Stone_Radius2 - 5))
        screen.blit(text_PCplayer, (Text_X + Stone_Radius2 * 3, Text_Y + Line_Space * 1.5 - Stone_Radius2 - 5))
        # 游戏结束
        text_end = my_font_small.render("Game end.", True, color[0])
        if a == len(poses):
            continue
        pos = poses[a]
        b += 1
        for ele in pos[0:b]:
            [i, j] = ele[1]
            x = Start_X + SIZE * i
            y = Start_Y + SIZE * j
            pygame.draw.circle(screen, color[ele[0] - 1], [x, y], Stone_Radius)

        pygame.display.flip()

        if b == len(pos):
            b = 0
            screen.blit(text_end, (Text_X, Text_Y + Line_Space * 2))
            if winners[a] == "Tie":
                text_winner = my_font_small.render("Tie.", True, color[0])
            else:
                text_winner = my_font_small.render("The winner is " + winners[a], True, color[0])
            screen.blit(text_winner, (Text_X, Text_Y + Line_Space * 2.5))
            a += 1
            pygame.display.flip()
            time.sleep(5)

        time.sleep(0.3)

    pygame.quit()


# 画棋子

if __name__ == '__main__':
    poses = [[(1, [3, 3]), (2, [1, 5])], [(1, [2, 5]), (2, [8, 3])]]
    winners = ["Tie", "MCTS_Pure"]
    _draw(poses, winners, Line_Points)

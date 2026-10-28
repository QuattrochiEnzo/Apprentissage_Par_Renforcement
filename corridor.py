from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional
import copy
from collections import deque

Position = Tuple[int, int]
# Actions:
#  - ("M", (r, c))                   : déplacer le pion courant sur la case (r,c)
#  - ("W", (r, c, "H"|"V"))          : placer un mur coin sup-gauche (r,c), horizontal ou vertical
Action = Tuple[str, Tuple[int, int]] | Tuple[str, Tuple[int, int, str]]

@dataclass
class Corridor:
    """
    Implémentation pédagogique du jeu Corridor/Quoridor pour 2 joueurs.

    Plateau:
      - taille N x N (par défaut 9)
      - P1 démarre en (0, N//2) et vise la ligne N-1
      - P2 démarre en (N-1, N//2) et vise la ligne 0
      - Chaque joueur possède W murs (par défaut 10)

    API (type Gym):
      - reset() -> observation
      - legal_actions() -> List[Action]
      - step(action) -> (observation, reward, done, info)
      - render()
      - clone()
      - shortest_path_length(player)

    Reward:
      - 0 sur les coups non terminaux
      - +1 pour le gagnant au moment du coup gagnant
    """
    N: int = 9
    walls_per_player: int = 10

    p1: Position = field(init=False)
    p2: Position = field(init=False)
    to_play: int = field(init=False, default=1)  # 1 ou 2
    walls_left: Dict[int, int] = field(init=False)
    H: Set[Tuple[int, int]] = field(init=False, default_factory=set)  # murs horizontaux
    V: Set[Tuple[int, int]] = field(init=False, default_factory=set)  # murs verticaux
    move_count: int = field(init=False, default=0)

    def __post_init__(self):
        self.reset()

    # ---------- Interface ----------
    def reset(self) -> Dict:
        self.p1 = (0, self.N // 2)
        self.p2 = (self.N - 1, self.N // 2)
        self.to_play = 1
        self.walls_left = {1: self.walls_per_player, 2: self.walls_per_player}
        self.H.clear()
        self.V.clear()
        self.move_count = 0
        return self._observation()

    def legal_actions(self) -> List[Action]:
        return self.legal_moves(self.to_play) + self.legal_walls(self.to_play)

    def step(self, action: Action) -> Tuple[Dict, float, bool, Dict]:
        assert not self.is_terminal(), "Partie terminée."
        kind = action[0]
        if kind == "M":
            _, (r, c) = action
            assert (r, c) in [pos for _, pos in self.legal_moves(self.to_play)], "Déplacement illégal."
            self._apply_move((r, c))
        elif kind == "W":
            _, (r, c, ori) = action
            assert (r, c, ori) in [w for _, w in self.legal_walls(self.to_play)], "Mur illégal."
            self._place_wall((r, c), ori)
        else:
            raise ValueError("Action inconnue.")

        self.move_count += 1
        winner = self.winner()
        if winner is not None:
            reward = 1.0 if winner == self.to_play else -1.0
            return self._observation(), reward, True, {"winner": winner}

        self.to_play = 2 if self.to_play == 1 else 1
        return self._observation(), 0.0, False, {}

    def render(self):
        board = [[" . " for _ in range(self.N)] for __ in range(self.N)]
        r1, c1 = self.p1
        r2, c2 = self.p2
        board[r1][c1] = " P1"
        board[r2][c2] = " P2"
        print("\nTour:", self.to_play, "| Murs restants -> P1:", self.walls_left[1], "P2:", self.walls_left[2])
        for r in range(self.N):
            line = ""
            for c in range(self.N):
                line += board[r][c]
                if c < self.N - 1:
                    line += "|" if self._is_blocked_vertical((r, c), (r, c + 1)) else " "
            print(line)
            if r < self.N - 1:
                line = ""
                for c in range(self.N):
                    line += ("---" if self._is_blocked_horizontal((r, c), (r + 1, c)) else "   ")
                    if c < self.N - 1:
                        line += "+"
                print(line)
        print()

    def clone(self) -> "Corridor":
        return copy.deepcopy(self)

    # ---------- Terminaison ----------
    def is_terminal(self) -> bool:
        return self.winner() is not None

    def winner(self) -> Optional[int]:
        if self.p1[0] == self.N - 1:
            return 1
        if self.p2[0] == 0:
            return 2
        return None

    def current_pawn(self, player: int) -> Position:
        return self.p1 if player == 1 else self.p2

    def opponent_pawn(self, player: int) -> Position:
        return self.p2 if player == 1 else self.p1

    # ---------- Coups légaux ----------
    def legal_moves(self, player: int) -> List[Tuple[str, Position]]:
        me = self.current_pawn(player)
        opp = self.opponent_pawn(player)
        moves: Set[Position] = set()

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = me[0]+dr, me[1]+dc
            if not self._can_step(me, (nr, nc)):
                continue
            if (nr, nc) == opp:
                jr, jc = opp[0]+dr, opp[1]+dc
                if self._can_step(opp, (jr, jc)):
                    moves.add((jr, jc))
                else:
                    for sdr, sdc in self._diagonal_branches(dr, dc):
                        ar, ac = opp[0]+sdr, opp[1]+sdc
                        if self._can_step(me, (nr, nc)) and self._can_step(opp, (ar, ac), from_pos=opp):
                            if self._can_diagonal(me, opp, (ar, ac)):
                                moves.add((ar, ac))
            else:
                moves.add((nr, nc))

        moves = {(r, c) for (r, c) in moves if 0 <= r < self.N and 0 <= c < self.N and (r, c) != opp}
        return [("M", pos) for pos in sorted(moves)]

    def legal_walls(self, player: int) -> List[Tuple[str, Tuple[int, int, str]]]:
        if self.walls_left[player] <= 0:
            return []
        actions = []
        for r in range(self.N - 1):
            for c in range(self.N - 1):
                for ori in ("H", "V"):
                    if self.is_legal_wall((r, c), ori):
                        actions.append(("W", (r, c, ori)))
        return actions

    def is_legal_wall(self, corner: Tuple[int, int], ori: str) -> bool:
        r, c = corner
        if not (0 <= r < self.N - 1 and 0 <= c < self.N - 1):
            return False
        if ori == "H":
            if (r, c) in self.H or (r, c - 1) in self.H or (r, c + 1) in self.H:
                return False
            if (r, c) in self.V or (r + 1, c) in self.V:
                return False
        elif ori == "V":
            if (r, c) in self.V or (r - 1, c) in self.V or (r + 1, c) in self.V:
                return False
            if (r, c) in self.H or (r, c + 1) in self.H:
                return False
        else:
            return False

        # test de connectivité (virtuel)
        if ori == "H":
            self.H.add((r, c))
        else:
            self.V.add((r, c))
        ok = self._has_path(1) and self._has_path(2)
        if ori == "H":
            self.H.discard((r, c))
        else:
            self.V.discard((r, c))
        return ok

    # ---------- Application des coups ----------
    def _apply_move(self, dst: Position):
        if self.to_play == 1:
            self.p1 = dst
        else:
            self.p2 = dst

    def _place_wall(self, corner: Tuple[int, int], ori: str):
        if ori == "H":
            self.H.add(corner)
        else:
            self.V.add(corner)
        self.walls_left[self.to_play] -= 1

    # ---------- Mouvements & murs ----------
    def _can_step(self, src: Position, dst: Position, from_pos: Optional[Position]=None) -> bool:
        r1, c1 = src
        r2, c2 = dst
        if not (0 <= r2 < self.N and 0 <= c2 < self.N):
            return False
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
        if r1 == r2:
            left = (r1, min(c1, c2))
            return not self._is_blocked_vertical((r1, left[1]), (r1, left[1] + 1))
        else:
            up = (min(r1, r2), c1)
            return not self._is_blocked_horizontal((up[0], c1), (up[0] + 1, c1))

    def _is_blocked_vertical(self, a: Position, b: Position) -> bool:
        (r, c1), (_, c2) = a, b
        if abs(c1 - c2) != 1 or r < 0 or r >= self.N:
            return False
        c = min(c1, c2)
        return (r, c) in self.V

    def _is_blocked_horizontal(self, a: Position, b: Position) -> bool:
        (r1, c), (r2, _) = a, b
        if abs(r1 - r2) != 1 or c < 0 or c >= self.N:
            return False
        r = min(r1, r2)
        return (r, c) in self.H

    @staticmethod
    def _diagonal_branches(dr: int, dc: int) -> List[Tuple[int, int]]:
        if dr != 0:
            return [(0, -1), (0, 1)]
        else:
            return [(-1, 0), (1, 0)]

    def _can_diagonal(self, me: Position, opp: Position, diag: Position) -> bool:
        if not self._can_step(me, opp):
            return False
        return self._can_step(opp, diag, from_pos=opp)

    # ---------- Connectivité ----------
    def _has_path(self, player: int) -> bool:
        start = self.current_pawn(player)
        goal_row = self.N - 1 if player == 1 else 0
        return self._bfs_reaches_row(start, goal_row)

    def shortest_path_length(self, player: int) -> Optional[int]:
        start = self.current_pawn(player)
        goal_row = self.N - 1 if player == 1 else 0
        return self._bfs_distance_to_row(start, goal_row)

    def _bfs_reaches_row(self, start: Position, goal_row: int) -> bool:
        q = deque([start])
        seen = {start}
        while q:
            r, c = q.popleft()
            if r == goal_row:
                return True
            for nr, nc in self._neighbors((r, c)):
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return False

    def _bfs_distance_to_row(self, start: Position, goal_row: int) -> Optional[int]:
        q = deque([(start, 0)])
        seen = {start}
        while q:
            (r, c), d = q.popleft()
            if r == goal_row:
                return d
            for nr, nc in self._neighbors((r, c)):
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append(((nr, nc), d + 1))
        return None

    def _neighbors(self, pos: Position) -> List[Position]:
        r, c = pos
        neigh = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if self._can_step((r, c), (nr, nc)):
                neigh.append((nr, nc))
        return neigh

    # ---------- Observation ----------
    def _observation(self) -> Dict:
        return {
            "N": self.N,
            "to_play": self.to_play,
            "p1": self.p1,
            "p2": self.p2,
            "walls_left": dict(self.walls_left),
            "H": set(self.H),
            "V": set(self.V),
            "move_count": self.move_count,
        }